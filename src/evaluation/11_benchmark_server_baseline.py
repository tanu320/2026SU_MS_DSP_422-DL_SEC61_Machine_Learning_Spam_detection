"""Benchmark the current server/local inference pipeline on the Android eval manifest."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dagshub
import mlflow
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.evaluation.inference_pipeline import InferencePipeline
from src.utils.mlflow_reporting import log_classification_artifacts, log_dataframe_artifact, log_json_artifact


EXPERIMENT_NAME = "scam-detection/refactored_pipeline/11_server_android_eval_baseline"


def configure_tracking(skip_mlflow: bool) -> bool:
    if skip_mlflow:
        return False
    import os

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
        print("[WARN] Missing DagsHub repo env vars; MLflow logging disabled.")
        return False

    dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return True


def normalize_prediction(value):
    if isinstance(value, str):
        lowered = value.lower()
        if "scam" in lowered or "spam" in lowered or "fraud" in lowered:
            return 1
        if "legit" in lowered or "safe" in lowered or "ham" in lowered:
            return 0
    return int(value)


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((pct / 100) * (len(values) - 1))))
    return float(values[index])


def summarize(rows):
    y_true = [int(row["true_label"]) for row in rows if row.get("error") == ""]
    y_pred = [int(row["predicted_label"]) for row in rows if row.get("error") == ""]
    total_latencies = [float(row["total_latency_sec"]) for row in rows if row.get("error") == ""]
    asr_latencies = [float(row["asr_latency_sec"]) for row in rows if row.get("error") == ""]
    clf_latencies = [float(row["classifier_latency_sec"]) for row in rows if row.get("error") == ""]
    durations = [float(row["duration_sec"]) for row in rows if row.get("duration_sec")]

    metrics = {
        "evaluated_rows": len(y_true),
        "failed_rows": len(rows) - len(y_true),
        "accuracy": accuracy_score(y_true, y_pred) if y_true else 0.0,
        "precision": precision_score(y_true, y_pred, zero_division=0) if y_true else 0.0,
        "recall": recall_score(y_true, y_pred, zero_division=0) if y_true else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0) if y_true else 0.0,
        "latency_mean_sec": statistics.mean(total_latencies) if total_latencies else 0.0,
        "latency_p50_sec": percentile(total_latencies, 50),
        "latency_p95_sec": percentile(total_latencies, 95),
        "asr_latency_mean_sec": statistics.mean(asr_latencies) if asr_latencies else 0.0,
        "classifier_latency_mean_sec": statistics.mean(clf_latencies) if clf_latencies else 0.0,
        "time_to_actionable_insight_mean_sec": statistics.mean(total_latencies) if total_latencies else 0.0,
        "real_time_factor_mean": (
            statistics.mean([lat / dur for lat, dur in zip(total_latencies, durations) if dur > 0])
            if durations
            else 0.0
        ),
    }
    if y_true:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics, y_true, y_pred


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/android_eval_manifest.csv")
    parser.add_argument("--config", default="configs/best_inference_config.json")
    parser.add_argument("--output_csv", default="android_server_baseline_results.csv")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--skip_mlflow", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    pipeline = InferencePipeline(args.config)

    rows = []
    for index, row in manifest.iterrows():
        audio_path = Path(row["audio_path"])
        print(f"[{index + 1}/{len(manifest)}] {row['sample_id']} -> {audio_path.name}")
        record = {
            "sample_id": row["sample_id"],
            "audio_filename": row["audio_filename"],
            "true_label": int(row["label"]),
            "duration_sec": row.get("duration_sec", ""),
            "error": "",
        }
        start = time.time()
        try:
            result = pipeline.process_audio(str(audio_path))
            prediction = normalize_prediction(result["prediction"])
            metrics = result.get("metrics", {})
            record.update(
                {
                    "predicted_label": prediction,
                    "prediction_raw": result["prediction"],
                    "transcript": result.get("transcript", ""),
                    "asr_latency_sec": metrics.get("asr_latency", 0.0),
                    "classifier_latency_sec": metrics.get("classifier_latency", 0.0),
                    "total_latency_sec": metrics.get("total_latency", time.time() - start),
                    "time_to_actionable_insight_sec": metrics.get("total_latency", time.time() - start),
                }
            )
        except Exception as exc:
            record.update(
                {
                    "predicted_label": -1,
                    "prediction_raw": "",
                    "transcript": "",
                    "asr_latency_sec": 0.0,
                    "classifier_latency_sec": 0.0,
                    "total_latency_sec": time.time() - start,
                    "time_to_actionable_insight_sec": 0.0,
                    "error": repr(exc),
                }
            )
        rows.append(record)

    results = pd.DataFrame(rows)
    results.to_csv(args.output_csv, index=False)
    summary, y_true, y_pred = summarize(rows)
    print(json.dumps(summary, indent=2))

    if configure_tracking(args.skip_mlflow):
        with mlflow.start_run(run_name="server_baseline_on_android_eval_manifest"):
            mlflow.set_tag("pipeline_stage", "11_server_android_eval_baseline")
            mlflow.log_param("manifest", args.manifest)
            mlflow.log_param("config", args.config)
            mlflow.log_artifact(args.manifest, artifact_path="manifest")
            log_dataframe_artifact(results, Path(args.output_csv).name, "predictions")
            log_json_artifact(summary, "server_baseline_summary.json", "metrics")
            for key, value in summary.items():
                mlflow.log_metric(key, float(value))
            if y_true:
                log_classification_artifacts(
                    y_true,
                    y_pred,
                    artifact_path="evaluation",
                    prefix="server_android_eval_baseline",
                )


if __name__ == "__main__":
    main()
