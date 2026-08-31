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


def configure_tracking(skip_mlflow: bool) -> bool:
    if skip_mlflow:
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
        print("[WARN] DagsHub env vars are missing; MLflow logging disabled.")
        return False

    dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return True


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


def prepare_qat_model(model_dir: str):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=2,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    ).float()
    model.train()

    model.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
    for module in model.modules():
        if isinstance(module, (torch.nn.Embedding, torch.nn.LayerNorm)):
            module.qconfig = None

    torch.quantization.prepare_qat(model, inplace=True)
    return model


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


def export_int8_onnx(model, tokenizer, output_dir: Path, max_length: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / "modernbert_qat_fp32.onnx"
    int8_path = output_dir / "modernbert_qat_int8.onnx"

    model.eval()
    model.apply(torch.ao.quantization.disable_fake_quant)
    wrapped = LogitsWrapper(model).cpu().eval()

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
    parser.add_argument("--model_dir", default="scam-classifier-model-transcript-lora")
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
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Expected a local Hugging Face model directory at {model_dir}. "
            "Fetch the transcript-trained model from DagsHub first or pass --model_dir."
        )

    tracking_enabled = configure_tracking(args.skip_mlflow)
    calibration_df = load_calibration_frame(args.calibration_data, args.calibration_rows)
    ensure_processed_data([args.eval_data], allow_rebuild=True)
    eval_df = pd.read_csv(args.eval_data)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = prepare_qat_model(str(model_dir))
    dataset = tokenize_frame(calibration_df, tokenizer, args.max_length)
    qat_metrics = run_qat_loop(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        lr=args.lr,
    )
    export_metrics = export_int8_onnx(model, tokenizer, Path(args.output_dir), args.max_length)
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
                "calibration_data": args.calibration_data,
                "eval_data": args.eval_data,
                "max_length": args.max_length,
                "calibration_rows": len(calibration_df),
                "eval_rows": min(args.eval_rows, len(eval_df)),
                "quantization": "QAT-assisted dynamic INT8 ONNX",
                "onnx_opset": 17,
                "onnxruntime_provider_target": "NNAPI with CPU fallback",
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
