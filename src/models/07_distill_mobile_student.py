"""Distill the transcript-trained ModernBERT teacher into a compact student.

The main project deployment remains the GGUF/GGML Hugging Face CPU path. This
script creates a smaller mobile/CPU candidate without relying on the failed
ModernBERT QAT -> ONNX path:

    teacher: transcript-trained ModernBERT
    student: MiniLM or MobileBERT sequence classifier
    loss: alpha * supervised CE + (1 - alpha) * temperature-scaled KL

Data usage is intentionally conservative:
- training uses data/processed/global_train.csv filtered to spoken ASR by default
- validation uses data/processed/global_val.csv filtered to spoken ASR
- global_test remains untouched unless explicitly passed later for final eval
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import dagshub
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_recall_fscore_support
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_access import ensure_processed_data
from src.utils.mlflow_reporting import (
    log_classification_artifacts,
    log_split_profile,
    log_training_history,
    log_transformer_model_with_fallback,
)


PIPELINE_STAGE = "07_mobile_student_distillation"
EXPERIMENT_NAME = "scam-detection/refactored_pipeline/07_mobile_student_distillation"
DEFAULT_TEACHER_EXPERIMENT = "scam-detection/refactored_pipeline/05_transcript_modernbert"
STUDENT_CHOICES = {
    "minilm": "microsoft/MiniLM-L12-H384-uncased",
    "mobilebert": "google/mobilebert-uncased",
}


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def configure_mlflow_for_rank() -> None:
    if is_main_process():
        return

    def noop(*args, **kwargs):
        return None

    mlflow.set_experiment = noop
    mlflow.log_artifact = noop
    mlflow.log_artifacts = noop
    mlflow.set_tag = noop
    mlflow.log_param = noop
    mlflow.log_params = noop
    mlflow.log_metric = noop
    mlflow.log_metrics = noop
    mlflow.start_run = lambda *args, **kwargs: nullcontext()
    if hasattr(mlflow, "transformers"):
        mlflow.transformers.log_model = noop


def configure_tracking(skip_mlflow: bool) -> bool:
    if skip_mlflow or not is_main_process():
        return False

    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if username and password:
        os.environ["DAGSHUB_USER"] = username
        os.environ["DAGSHUB_TOKEN"] = password
        dagshub.auth.add_app_token(password)

    if not repo_owner or not repo_name:
        print("[WARN] DagsHub env vars missing; MLflow logging disabled.")
        return False

    dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return True


def configure_dagshub_for_artifacts() -> None:
    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if username and password:
        os.environ["DAGSHUB_USER"] = username
        os.environ["DAGSHUB_TOKEN"] = password
        dagshub.auth.add_app_token(password)

    if not repo_owner or not repo_name:
        raise RuntimeError(
            "DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME are required to fetch teacher artifacts."
        )

    dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)


def latest_finished_run_id(experiment_name: str) -> str:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=20,
    )
    if runs.empty:
        raise RuntimeError(f"No finished runs found in MLflow experiment: {experiment_name}")

    if "metrics.eval_f1" in runs.columns:
        scored = runs[runs["metrics.eval_f1"].notna()]
        if not scored.empty:
            return str(scored.sort_values("start_time", ascending=False).iloc[0].run_id)
    return str(runs.iloc[0].run_id)


def find_artifact_paths(client, run_id: str, target_names: set[str], base_path: str = "") -> list[str]:
    matches = []
    for artifact in client.list_artifacts(run_id, path=base_path):
        if artifact.is_dir:
            matches.extend(find_artifact_paths(client, run_id, target_names, artifact.path))
        elif Path(artifact.path).name in target_names:
            matches.append(artifact.path)
    return matches


def score_model_dir(path: Path) -> int:
    files = {child.name for child in path.iterdir() if child.is_file()}
    score = 0
    if "config.json" in files and "model.safetensors" in files:
        score += 100
    if path.name == "hf_model_artifacts":
        score += 25
    if "tokenizer.json" in files or "tokenizer_config.json" in files:
        score += 10
    if path.name.startswith("checkpoint-"):
        score -= 20
    return score


def discover_model_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    candidates = []
    for dirpath, _, filenames in os.walk(root):
        files = set(filenames)
        if "config.json" in files and "model.safetensors" in files:
            candidates.append(Path(dirpath))
    return sorted(candidates, key=score_model_dir, reverse=True)


def resolve_teacher_dir(args) -> Path:
    teacher_dir = Path(args.teacher_model_dir)
    if (teacher_dir / "config.json").exists() and (teacher_dir / "model.safetensors").exists():
        print(f"Using local teacher model: {teacher_dir}")
        return teacher_dir
    if teacher_dir.exists():
        nested = discover_model_dirs(teacher_dir)
        if nested:
            print(f"Using nested teacher model: {nested[0]}")
            return nested[0]

    cache = Path(args.teacher_download_dir)
    cached = discover_model_dirs(cache)
    if cached and not args.teacher_run_id:
        print(f"Using cached teacher model: {cached[0]}")
        return cached[0]

    configure_dagshub_for_artifacts()

    run_id = args.teacher_run_id or latest_finished_run_id(args.teacher_experiment)
    client = mlflow.tracking.MlflowClient()
    artifacts = find_artifact_paths(client, run_id, {"model.safetensors", "config.json"})
    if not artifacts:
        raise RuntimeError(f"No Hugging Face teacher model artifacts found in run {run_id}.")

    roots = list(dict.fromkeys(str(Path(path).parent) for path in artifacts))
    roots = sorted(
        roots,
        key=lambda path: (path.endswith("hf_model_artifacts"), "/checkpoint-" not in path),
        reverse=True,
    )
    cache.mkdir(parents=True, exist_ok=True)
    last_error = None
    for root in roots:
        try:
            print(f"Downloading teacher artifact root: runs:/{run_id}/{root}")
            local_root = Path(
                mlflow.artifacts.download_artifacts(
                    artifact_uri=f"runs:/{run_id}/{root}",
                    dst_path=str(cache),
                )
            )
            discovered = discover_model_dirs(local_root)
            if discovered:
                print(f"Resolved teacher model: {discovered[0]}")
                return discovered[0]
        except Exception as exc:
            last_error = exc
            print(f"[WARN] Could not download teacher artifact root {root}: {exc}")

    raise RuntimeError(f"Could not resolve teacher model from run {run_id}. Last error: {last_error}")


def load_split(path: str, source_domain: str | None, max_rows: int | None, seed: int) -> pd.DataFrame:
    ensure_processed_data([path], allow_rebuild=True)
    df = pd.read_csv(path)
    if source_domain and "source_domain" in df.columns:
        df = df[df["source_domain"] == source_domain].copy()
    if max_rows and len(df) > max_rows:
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda group: group.sample(frac=max_rows / len(df), random_state=seed))
            .sample(frac=1.0, random_state=seed)
            .head(max_rows)
            .reset_index(drop=True)
        )
    if df.empty:
        raise ValueError(f"No rows available after filtering {path} with source_domain={source_domain!r}.")
    return df.reset_index(drop=True)


def compute_teacher_logits(df: pd.DataFrame, teacher_model, teacher_tokenizer, device: str, max_length: int, batch_size: int) -> np.ndarray:
    teacher_model.to(device)
    teacher_model.eval()
    logits = []
    texts = df["text"].astype(str).tolist()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            batch = teacher_tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = teacher_model(**batch).logits.detach().cpu().numpy()
            logits.append(outputs)
            if start == 0 or (start + len(batch_texts)) % 500 == 0 or start + len(batch_texts) == len(texts):
                print(f"Teacher logits: {min(start + len(batch_texts), len(texts))}/{len(texts)}")
    return np.concatenate(logits, axis=0)


def build_dataset(df: pd.DataFrame, teacher_logits: np.ndarray, tokenizer, max_length: int) -> Dataset:
    payload = df[["text", "label"]].rename(columns={"label": "labels"}).reset_index(drop=True)
    payload["teacher_logits"] = teacher_logits.tolist()
    dataset = Dataset.from_pandas(payload)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset = dataset.remove_columns(["text"])
    dataset.set_format("torch")
    return dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    if isinstance(labels, tuple):
        labels = labels[0]
    preds = np.argmax(logits, axis=-1)
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    pos_probs = probs[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": average_precision_score(labels, pos_probs),
    }


class DistillationTrainer(Trainer):
    def __init__(self, *args, temperature: float = 2.0, alpha: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        teacher_logits = inputs.pop("teacher_logits")
        labels = inputs.get("labels")
        outputs = model(**inputs)
        supervised_loss = F.cross_entropy(outputs.logits, labels)
        student_log_probs = F.log_softmax(outputs.logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (self.temperature ** 2)
        loss = (self.alpha * supervised_loss) + ((1.0 - self.alpha) * distill_loss)
        return (loss, outputs) if return_outputs else loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="data/processed/global_train.csv")
    parser.add_argument("--eval_data", default="data/processed/global_val.csv")
    parser.add_argument(
        "--source_domain",
        default="spoken_asr",
        help="Default distills the final transcript-trained behavior. Use written_text for a text-only student.",
    )
    parser.add_argument("--teacher_model_dir", default="scam-classifier-model-transcript-lora")
    parser.add_argument("--teacher_run_id", default=None)
    parser.add_argument("--teacher_experiment", default=DEFAULT_TEACHER_EXPERIMENT)
    parser.add_argument("--teacher_download_dir", default="downloads/mobile_student_teacher")
    parser.add_argument(
        "--student_model_name",
        default=STUDENT_CHOICES["minilm"],
        help=f"HF student checkpoint. Shortcuts: {', '.join(STUDENT_CHOICES)}.",
    )
    parser.add_argument("--output_dir", default="scam-classifier-model-mobile-student")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5, help="Supervised CE weight; distillation weight is 1-alpha.")
    parser.add_argument("--teacher_batch_size", type=int, default=16)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_mlflow", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.student_model_name in STUDENT_CHOICES:
        args.student_model_name = STUDENT_CHOICES[args.student_model_name]

    configure_mlflow_for_rank()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_seed(args.seed)

    train_df = load_split(args.train_data, args.source_domain, args.max_train_samples, args.seed)
    eval_df = load_split(args.eval_data, args.source_domain, args.max_eval_samples, args.seed)
    print(f"Distillation train rows: {len(train_df)} | eval rows: {len(eval_df)} | source_domain={args.source_domain}")

    tracking_enabled = configure_tracking(args.skip_mlflow)
    teacher_dir = resolve_teacher_dir(args)

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_dir)
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        teacher_dir,
        num_labels=2,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    print(f"Loaded teacher from {teacher_dir}")

    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model_name)
    student_config = AutoConfig.from_pretrained(args.student_model_name, num_labels=2)
    student_model = AutoModelForSequenceClassification.from_pretrained(args.student_model_name, config=student_config)

    train_teacher_logits = compute_teacher_logits(
        train_df,
        teacher_model,
        teacher_tokenizer,
        args.device,
        args.max_length,
        args.teacher_batch_size,
    )
    eval_teacher_logits = compute_teacher_logits(
        eval_df,
        teacher_model,
        teacher_tokenizer,
        args.device,
        args.max_length,
        args.teacher_batch_size,
    )
    teacher_model.cpu()

    train_ds = build_dataset(train_df, train_teacher_logits, student_tokenizer, args.max_length)
    eval_ds = build_dataset(eval_df, eval_teacher_logits, student_tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=student_tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],
        max_steps=args.max_steps,
        seed=args.seed,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=student_tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        temperature=args.temperature,
        alpha=args.alpha,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(f"Final student evaluation metrics: {metrics}")

    predictions = trainer.predict(eval_ds)
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=-1)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    trainer.save_model(args.output_dir)
    student_tokenizer.save_pretrained(args.output_dir)
    print(f"Student model saved to {args.output_dir}")

    if tracking_enabled and is_main_process():
        with mlflow.start_run(run_name="07-mobile-student-distillation"):
            mlflow.set_tag("pipeline_stage", PIPELINE_STAGE)
            mlflow.set_tag("model_family", "teacher_student_distillation")
            mlflow.log_params({
                "teacher_model_dir": str(teacher_dir),
                "teacher_run_id": args.teacher_run_id or "",
                "teacher_experiment": args.teacher_experiment,
                "student_model_name": args.student_model_name,
                "source_domain": args.source_domain or "",
                "train_data": args.train_data,
                "eval_data": args.eval_data,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "max_length": args.max_length,
                "temperature": args.temperature,
                "alpha": args.alpha,
                "distillation_loss": "alpha*CE + (1-alpha)*KL(student||teacher)",
            })
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))
            log_split_profile({"train": train_df, "eval": eval_df})
            log_classification_artifacts(y_true, y_pred, artifact_path="evaluation", prefix="mobile_student")
            log_training_history(trainer.state.log_history, artifact_path="training", prefix="mobile_student")
            log_transformer_model_with_fallback(
                components={"model": student_model, "tokenizer": student_tokenizer},
                output_dir=args.output_dir,
                artifact_path=PIPELINE_STAGE,
                registered_model_name="Mobile-Student-Scam-Classifier",
                task="text-classification",
            )


if __name__ == "__main__":
    main()
