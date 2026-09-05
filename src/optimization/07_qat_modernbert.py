"""QAT-assisted Android export for the transcript-adapted ModernBERT classifier.

This script is intentionally separate from the GGUF/whisper.cpp CPU-serving path.
It prepares an Android-friendly ONNX classifier that preserves the sequence
classification head directly:

    input_ids + attention_mask -> logits[legitimate, scam]

The QAT loop uses only data/processed/ptq_calibration.csv, which is sampled from
the training split by src/data/02_build_datasets.py and validated to be disjoint
from validation/test data by src/data/03_validate_partitions.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dagshub
import mlflow
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from src.utils.data_access import ensure_processed_data
from src.utils.mlflow_reporting import log_dataframe_artifact, log_json_artifact


EXPERIMENT_NAME = "scam-detection/refactored_pipeline/07_android_qat_export"
DEFAULT_MODEL_EXPERIMENT = "scam-detection/refactored_pipeline/05_transcript_modernbert"
DEFAULT_BASE_MODEL_EXPERIMENT = "scam-detection/refactored_pipeline/04_universal_modernbert"


def configure_dagshub_mlflow(experiment_name: str | None = None) -> bool:
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
        print("[WARN] DagsHub env vars are missing; MLflow logging disabled.")
        return False

    dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    return True


def configure_tracking(skip_mlflow: bool) -> bool:
    if skip_mlflow:
        return False
    return configure_dagshub_mlflow(EXPERIMENT_NAME)


def find_artifact_paths(client, run_id: str, target_names: set[str], base_path: str = "") -> list[str]:
    matches = []
    for artifact in client.list_artifacts(run_id, path=base_path):
        if artifact.is_dir:
            matches.extend(find_artifact_paths(client, run_id, target_names, artifact.path))
        elif Path(artifact.path).name in target_names:
            matches.append(artifact.path)
    return matches


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


def score_model_dir(path: Path) -> int:
    files = {child.name for child in path.iterdir() if child.is_file()}
    score = 0
    if "config.json" in files and "model.safetensors" in files:
        score += 100
    if path.name == "hf_model_artifacts":
        score += 25
    if "adapter_config.json" in files and "adapter_model.safetensors" in files:
        score += 50
    if "tokenizer.json" in files or "tokenizer_config.json" in files:
        score += 10
    if path.name.startswith("checkpoint-"):
        score -= 20
    return score


def discover_model_dirs(root: Path) -> list[Path]:
    candidates = []
    if not root.exists():
        return candidates
    for dirpath, _, filenames in os.walk(root):
        files = set(filenames)
        if "config.json" in files and "model.safetensors" in files:
            candidates.append(Path(dirpath))
        elif "adapter_config.json" in files and "adapter_model.safetensors" in files:
            candidates.append(Path(dirpath))
    return sorted(candidates, key=score_model_dir, reverse=True)


def resolve_model_dir(
    preferred_model_dir: str,
    model_run_id: str | None,
    model_experiment: str,
    download_dir: str,
    use_download_cache: bool = True,
) -> Path:
    model_dir = Path(preferred_model_dir)
    if (model_dir / "config.json").exists() or (model_dir / "adapter_config.json").exists():
        print(f"Using local model directory: {model_dir}")
        return model_dir

    if model_dir.exists():
        nested_models = discover_model_dirs(model_dir)
        if nested_models:
            print(f"Using nested model directory under requested path: {nested_models[0]}")
            return nested_models[0]

    search_caches = list(dict.fromkeys([Path(download_dir), Path("downloads")]))
    prefer_remote_run = model_run_id is not None
    if use_download_cache and not prefer_remote_run:
        existing_downloads = []
        for cache_root in search_caches:
            existing_downloads.extend(discover_model_dirs(cache_root))
        existing_downloads = sorted(list(dict.fromkeys(existing_downloads)), key=score_model_dir, reverse=True)
        if existing_downloads:
            print(f"Using model already available in download cache: {existing_downloads[0]}")
            return existing_downloads[0]

    print(f"Local model directory not found: {model_dir}")
    print("Attempting to fetch transcript-trained model from DagsHub MLflow artifacts...")
    if not configure_dagshub_mlflow(None):
        raise RuntimeError(
            "DagsHub/MLflow credentials are required to fetch a missing model. "
            "Set DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME, MLFLOW_TRACKING_USERNAME, "
            "and MLFLOW_TRACKING_PASSWORD."
        )

    run_id = model_run_id or latest_finished_run_id(model_experiment)
    client = mlflow.tracking.MlflowClient()
    model_artifacts = find_artifact_paths(
        client,
        run_id,
        {"model.safetensors", "adapter_model.safetensors", "config.json", "adapter_config.json"},
    )
    model_artifacts = [path for path in model_artifacts if path]
    if not model_artifacts:
        raise RuntimeError(
            f"No Hugging Face model or LoRA adapter artifacts found in run {run_id}. "
            "Pass --model_run_id explicitly if the latest transcript run is not the model run."
        )

    candidate_roots = []
    for artifact_path in model_artifacts:
        candidate_roots.append(str(Path(artifact_path).parent))
    # Preserve order while removing duplicates.
    candidate_roots = list(dict.fromkeys(candidate_roots))

    def remote_root_score(path: str) -> int:
        score = 0
        if path.endswith("hf_model_artifacts"):
            score += 30
        if "/checkpoint-" in path or path.startswith("checkpoint-"):
            score -= 30
        if path.endswith("adapter_model.safetensors"):
            score -= 10
        return score

    candidate_roots = sorted(candidate_roots, key=remote_root_score, reverse=True)

    download_root = Path(download_dir)
    download_root.mkdir(parents=True, exist_ok=True)
    last_error = None
    for artifact_root in candidate_roots:
        try:
            print(f"Downloading model artifact root: runs:/{run_id}/{artifact_root}")
            local_root = Path(
                mlflow.artifacts.download_artifacts(
                    artifact_uri=f"runs:/{run_id}/{artifact_root}",
                    dst_path=str(download_root),
                )
            )
            discovered = discover_model_dirs(local_root)
            if discovered:
                resolved = discovered[0]
                if (resolved / "config.json").exists() and (resolved / "model.safetensors").exists():
                    print(f"Resolved merged Hugging Face model directory: {resolved}")
                else:
                    print(f"Resolved LoRA adapter directory: {resolved}")
                return resolved
        except Exception as exc:
            last_error = exc
            print(f"[WARN] Could not download artifact root {artifact_root}: {exc}")

    if use_download_cache and not prefer_remote_run:
        discovered = []
        for cache_root in search_caches + [download_root]:
            discovered.extend(discover_model_dirs(cache_root))
        discovered = sorted(list(dict.fromkeys(discovered)), key=score_model_dir, reverse=True)
        if discovered:
            print(f"Resolved model directory from downloaded artifact cache: {discovered[0]}")
            return discovered[0]

    raise RuntimeError(
        f"Failed to resolve a usable model directory from run {run_id}. Last error: {last_error}"
    )


def resolve_relative_base_model(
    base_model_name: str,
    adapter_dir: Path,
    base_model_dir: str | None = None,
    base_model_run_id: str | None = None,
    base_model_experiment: str = DEFAULT_BASE_MODEL_EXPERIMENT,
    download_dir: str = "downloads/android_qat_base_model",
    allow_public_base_fallback: bool = False,
) -> str:
    if not base_model_name.startswith("."):
        return base_model_name

    if base_model_dir:
        explicit_base = Path(base_model_dir)
        if (explicit_base / "config.json").exists() or (explicit_base / "adapter_config.json").exists():
            print(f"Using explicit LoRA base model directory: {explicit_base}")
            return str(explicit_base)
        nested_base = discover_model_dirs(explicit_base)
        if nested_base:
            print(f"Using nested explicit LoRA base model directory: {nested_base[0]}")
            return str(nested_base[0])
        raise FileNotFoundError(f"--base_model_dir was provided but no model files were found under {explicit_base}")

    direct = (Path.cwd() / base_model_name).resolve()
    if (direct / "config.json").exists() or (direct / "adapter_config.json").exists():
        return str(direct)
    if direct.exists():
        nested_direct = discover_model_dirs(direct)
        if nested_direct:
            print(f"Resolved nested local LoRA base model '{base_model_name}' -> {nested_direct[0]}")
            return str(nested_direct[0])

    target_name = Path(base_model_name).name
    search_roots = [
        adapter_dir.parent,
        adapter_dir.parents[1] if len(adapter_dir.parents) > 1 else adapter_dir.parent,
        Path("downloads"),
        Path("."),
    ]
    matches = []
    for root in search_roots:
        if not root.exists():
            continue
        for candidate in discover_model_dirs(root):
            if candidate.name == target_name or target_name in str(candidate):
                matches.append(candidate)

    if matches:
        matches = sorted(matches, key=score_model_dir, reverse=True)
        print(f"Resolved relative LoRA base model '{base_model_name}' -> {matches[0]}")
        return str(matches[0])

    if base_model_experiment:
        print(
            f"Could not resolve local LoRA base '{base_model_name}'. "
            f"Attempting to fetch it from MLflow experiment: {base_model_experiment}"
        )
        try:
            fetched_base = resolve_model_dir(
                preferred_model_dir=base_model_name,
                model_run_id=base_model_run_id,
                model_experiment=base_model_experiment,
                download_dir=download_dir,
                use_download_cache=False,
            )
            print(f"Resolved LoRA base model from MLflow: {fetched_base}")
            return str(fetched_base)
        except Exception as exc:
            print(f"[WARN] Could not fetch LoRA base model from MLflow: {exc}")

    if allow_public_base_fallback:
        fallback = "answerdotai/ModernBERT-base"
        print(
            f"[WARN] Could not resolve local LoRA base '{base_model_name}'. "
            f"Falling back to {fallback} because --allow_public_base_fallback was set."
        )
        return fallback

    raise FileNotFoundError(
        f"LoRA adapter expects local base model '{base_model_name}', but it was not found. "
        "This usually means the downloaded checkpoint is an adapter-only artifact. "
        "Pass a merged model directory via --model_dir, provide the universal base model artifacts, "
        "or rerun with --allow_public_base_fallback only for a non-final smoke test."
    )


def load_calibration_frame(path: str, rows: int) -> pd.DataFrame:
    ensure_processed_data([path], allow_rebuild=True)
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must contain text and label columns.")

    if rows and len(df) > rows:
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda group: group.sample(frac=rows / len(df), random_state=42))
            .sample(frac=1.0, random_state=42)
            .head(rows)
            .reset_index(drop=True)
        )
    return df.reset_index(drop=True)


def prepare_qat_model(model_dir: str, allow_public_base_fallback: bool):
    model = load_clean_model(
        model_dir,
        allow_public_base_fallback=allow_public_base_fallback,
    )
    model.train()

    model.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
    for module in model.modules():
        if isinstance(module, (torch.nn.Embedding, torch.nn.LayerNorm)):
            module.qconfig = None

    torch.quantization.prepare_qat(model, inplace=True)
    return model


def load_clean_model(
    model_dir: str,
    allow_public_base_fallback: bool = False,
    base_model_dir: str | None = None,
    base_model_run_id: str | None = None,
    base_model_experiment: str = DEFAULT_BASE_MODEL_EXPERIMENT,
    base_download_dir: str = "downloads/android_qat_base_model",
):
    model_path = Path(model_dir)
    if (model_path / "adapter_config.json").exists():
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:
            raise ImportError("Downloaded model is a LoRA adapter; install peft to merge it.") from exc

        peft_config = PeftConfig.from_pretrained(str(model_path))
        base_model_name = resolve_relative_base_model(
            peft_config.base_model_name_or_path,
            model_path,
            base_model_dir=base_model_dir,
            base_model_run_id=base_model_run_id,
            base_model_experiment=base_model_experiment,
            download_dir=base_download_dir,
            allow_public_base_fallback=allow_public_base_fallback,
        )
        print(f"Loading base model for LoRA merge: {base_model_name}")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=2,
            attn_implementation="eager",
            torch_dtype=torch.float32,
        )
        model = PeftModel.from_pretrained(base_model, str(model_path)).merge_and_unload()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            num_labels=2,
            attn_implementation="eager",
            torch_dtype=torch.float32,
        )

    return model.float()


def copy_qat_weights_to_clean_model(
    qat_model,
    model_dir: Path,
    allow_public_base_fallback: bool,
    base_model_dir: str | None = None,
    base_model_run_id: str | None = None,
    base_model_experiment: str = DEFAULT_BASE_MODEL_EXPERIMENT,
    base_download_dir: str = "downloads/android_qat_base_model",
):
    clean_model = load_clean_model(
        str(model_dir),
        allow_public_base_fallback=allow_public_base_fallback,
        base_model_dir=base_model_dir,
        base_model_run_id=base_model_run_id,
        base_model_experiment=base_model_experiment,
        base_download_dir=base_download_dir,
    )
    clean_state = clean_model.state_dict()
    qat_state = qat_model.cpu().state_dict()

    copied = 0
    compatible_state = {}
    for name, value in qat_state.items():
        if name not in clean_state:
            continue
        if clean_state[name].shape != value.shape:
            continue
        compatible_state[name] = value.detach().cpu()
        copied += 1

    missing, unexpected = clean_model.load_state_dict(compatible_state, strict=False)
    print(
        "Copied QAT-trained tensors into clean export model: "
        f"{copied} tensors | missing={len(missing)} | unexpected={len(unexpected)}"
    )
    clean_model.eval()
    return clean_model


def load_tokenizer(
    model_dir: Path,
    allow_public_base_fallback: bool = False,
    base_model_dir: str | None = None,
    base_model_run_id: str | None = None,
    base_model_experiment: str = DEFAULT_BASE_MODEL_EXPERIMENT,
    base_download_dir: str = "downloads/android_qat_base_model",
):
    if (model_dir / "tokenizer.json").exists() or (model_dir / "tokenizer_config.json").exists():
        return AutoTokenizer.from_pretrained(model_dir)

    if (model_dir / "adapter_config.json").exists():
        try:
            from peft import PeftConfig
        except ImportError as exc:
            raise ImportError("Downloaded model is a LoRA adapter; install peft to resolve tokenizer.") from exc

        peft_config = PeftConfig.from_pretrained(str(model_dir))
        base_model_name = resolve_relative_base_model(
            peft_config.base_model_name_or_path,
            model_dir,
            base_model_dir=base_model_dir,
            base_model_run_id=base_model_run_id,
            base_model_experiment=base_model_experiment,
            download_dir=base_download_dir,
            allow_public_base_fallback=allow_public_base_fallback,
        )
        return AutoTokenizer.from_pretrained(base_model_name)

    return AutoTokenizer.from_pretrained(model_dir)


def tokenize_frame(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_pandas(df[["text", "label"]].rename(columns={"label": "labels"}))

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset = dataset.remove_columns([c for c in dataset.column_names if c == "text"])
    dataset.set_format("torch")
    return dataset


def run_qat_loop(model, dataset: Dataset, tokenizer, device: torch.device, batch_size: int, lr: float):
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.to(device)
    model.train()
    losses = []
    start = time.time()

    for step, batch in enumerate(loader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))

        if step == 1 or step % 10 == 0 or step == len(loader):
            print(f"QAT step {step}/{len(loader)} | loss={losses[-1]:.4f}")

    return {
        "qat_steps": len(loader),
        "qat_train_seconds": time.time() - start,
        "qat_loss_last": losses[-1] if losses else 0.0,
        "qat_loss_mean": sum(losses) / len(losses) if losses else 0.0,
    }


class LogitsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export_int8_onnx(
    qat_model,
    model_dir: Path,
    tokenizer,
    output_dir: Path,
    max_length: int,
    allow_public_base_fallback: bool,
    base_model_dir: str | None = None,
    base_model_run_id: str | None = None,
    base_model_experiment: str = DEFAULT_BASE_MODEL_EXPERIMENT,
    base_download_dir: str = "downloads/android_qat_base_model",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / "modernbert_qat_fp32.onnx"
    int8_path = output_dir / "modernbert_qat_int8.onnx"

    clean_model = copy_qat_weights_to_clean_model(
        qat_model,
        model_dir,
        allow_public_base_fallback,
        base_model_dir=base_model_dir,
        base_model_run_id=base_model_run_id,
        base_model_experiment=base_model_experiment,
        base_download_dir=base_download_dir,
    )
    wrapped = LogitsWrapper(clean_model).cpu().eval()

    dummy = tokenizer(
        "This is a calibration trace for Android export.",
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=min(max_length, 512),
    )

    torch.onnx.export(
        wrapped,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(fp32_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
    )

    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul"],
    )

    tokenizer.save_pretrained(output_dir)
    manifest = {
        "classifier_model": int8_path.name,
        "fp32_reference_model": fp32_path.name,
        "tokenizer_files": ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
        "contract": {
            "inputs": ["input_ids:int64[batch,seq]", "attention_mask:int64[batch,seq]"],
            "outputs": ["logits:float32[batch,2]"],
            "scam_probability": "softmax(logits)[1]",
        },
        "android_runtime": "onnxruntime-android with NNAPI requested and CPU fallback",
    }
    (output_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "fp32_onnx_path": str(fp32_path),
        "int8_onnx_path": str(int8_path),
        "fp32_size_mb": fp32_path.stat().st_size / (1024 * 1024),
        "int8_size_mb": int8_path.stat().st_size / (1024 * 1024),
    }


def evaluate_onnx_smoke(onnx_path: str, tokenizer, df: pd.DataFrame, max_length: int, rows: int) -> dict:
    import numpy as np
    import onnxruntime as ort

    sample = df.sample(n=min(rows, len(df)), random_state=123).reset_index(drop=True)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    preds = []
    latencies = []

    for _, row in sample.iterrows():
        inputs = tokenizer(
            str(row["text"]),
            return_tensors="np",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        start = time.time()
        logits = session.run(None, {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        })[0]
        latencies.append(time.time() - start)
        preds.append(int(logits[0].argmax()))

    y_true = sample["label"].astype(int).tolist()
    return {
        "smoke_rows": len(sample),
        "smoke_accuracy": accuracy_score(y_true, preds),
        "smoke_f1": f1_score(y_true, preds, zero_division=0),
        "smoke_latency_ms_mean": 1000 * sum(latencies) / len(latencies) if latencies else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="scam-classifier-model-transcript-lora",
        help="Preferred local Hugging Face model directory. If missing, fetch from DagsHub MLflow.",
    )
    parser.add_argument(
        "--model_run_id",
        default=None,
        help="Optional MLflow run ID containing the transcript-trained model artifacts.",
    )
    parser.add_argument(
        "--model_experiment",
        default=DEFAULT_MODEL_EXPERIMENT,
        help="MLflow experiment to search when --model_dir is missing and --model_run_id is omitted.",
    )
    parser.add_argument("--download_dir", default="downloads/android_qat_model")
    parser.add_argument(
        "--base_model_dir",
        default=None,
        help="Optional local universal-model directory needed when the transcript artifact is a LoRA adapter.",
    )
    parser.add_argument(
        "--base_model_run_id",
        default=None,
        help="Optional MLflow run ID for the universal base model used by the transcript LoRA adapter.",
    )
    parser.add_argument(
        "--base_model_experiment",
        default=DEFAULT_BASE_MODEL_EXPERIMENT,
        help="MLflow experiment to search for the missing universal LoRA base model.",
    )
    parser.add_argument("--base_download_dir", default="downloads/android_qat_base_model")
    parser.add_argument("--calibration_data", default="data/processed/ptq_calibration.csv")
    parser.add_argument("--eval_data", default="data/processed/global_test.csv")
    parser.add_argument("--output_dir", default="models/android")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--calibration_rows", type=int, default=256)
    parser.add_argument("--eval_rows", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_mlflow", action="store_true")
    parser.add_argument(
        "--allow_public_base_fallback",
        action="store_true",
        help="Allow adapter-only smoke export by merging on answerdotai/ModernBERT-base if the local adapter base is missing. Do not use for final metrics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = resolve_model_dir(
        preferred_model_dir=args.model_dir,
        model_run_id=args.model_run_id,
        model_experiment=args.model_experiment,
        download_dir=args.download_dir,
    )

    tracking_enabled = configure_tracking(args.skip_mlflow)
    calibration_df = load_calibration_frame(args.calibration_data, args.calibration_rows)
    ensure_processed_data([args.eval_data], allow_rebuild=True)
    eval_df = pd.read_csv(args.eval_data)

    tokenizer = load_tokenizer(
        model_dir,
        allow_public_base_fallback=args.allow_public_base_fallback,
        base_model_dir=args.base_model_dir,
        base_model_run_id=args.base_model_run_id,
        base_model_experiment=args.base_model_experiment,
        base_download_dir=args.base_download_dir,
    )
    model = load_clean_model(
        str(model_dir),
        allow_public_base_fallback=args.allow_public_base_fallback,
        base_model_dir=args.base_model_dir,
        base_model_run_id=args.base_model_run_id,
        base_model_experiment=args.base_model_experiment,
        base_download_dir=args.base_download_dir,
    )
    model.train()

    model.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
    for module in model.modules():
        if isinstance(module, (torch.nn.Embedding, torch.nn.LayerNorm)):
            module.qconfig = None
    torch.quantization.prepare_qat(model, inplace=True)

    dataset = tokenize_frame(calibration_df, tokenizer, args.max_length)
    qat_metrics = run_qat_loop(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        lr=args.lr,
    )
    export_metrics = export_int8_onnx(
        model,
        model_dir,
        tokenizer,
        Path(args.output_dir),
        args.max_length,
        args.allow_public_base_fallback,
        base_model_dir=args.base_model_dir,
        base_model_run_id=args.base_model_run_id,
        base_model_experiment=args.base_model_experiment,
        base_download_dir=args.base_download_dir,
    )
    smoke_metrics = evaluate_onnx_smoke(
        export_metrics["int8_onnx_path"],
        tokenizer,
        eval_df,
        args.max_length,
        args.eval_rows,
    )

    metrics = {**qat_metrics, **export_metrics, **smoke_metrics}
    print(json.dumps(metrics, indent=2))

    if tracking_enabled:
        with mlflow.start_run(run_name="android_qat_int8_export"):
            mlflow.set_tag("pipeline_stage", "07_android_qat_export")
            mlflow.set_tag("android_model_contract", "input_ids_attention_mask_to_logits")
            mlflow.log_params({
                "model_dir": str(model_dir),
                "requested_model_dir": args.model_dir,
                "model_run_id": args.model_run_id or "",
                "model_experiment": args.model_experiment,
                "base_model_dir": args.base_model_dir or "",
                "base_model_run_id": args.base_model_run_id or "",
                "base_model_experiment": args.base_model_experiment,
                "calibration_data": args.calibration_data,
                "eval_data": args.eval_data,
                "max_length": args.max_length,
                "calibration_rows": len(calibration_df),
                "eval_rows": min(args.eval_rows, len(eval_df)),
                "quantization": "QAT-assisted dynamic INT8 ONNX",
                "onnx_opset": 17,
                "onnxruntime_provider_target": "NNAPI with CPU fallback",
                "allow_public_base_fallback": args.allow_public_base_fallback,
            })
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))
                else:
                    mlflow.log_param(key, value)
            log_dataframe_artifact(calibration_df, "android_qat_calibration.csv", "dataset")
            log_json_artifact(metrics, "android_qat_export_metrics.json", "metrics")
            mlflow.log_artifacts(args.output_dir, artifact_path="android_assets")


if __name__ == "__main__":
    main()
