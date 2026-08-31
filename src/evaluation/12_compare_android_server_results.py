"""Compare Android on-device benchmark output against the server/local baseline."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dagshub
import mlflow
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.utils.mlflow_reporting import log_dataframe_artifact, log_json_artifact


EXPERIMENT_NAME = "scam-detection/refactored_pipeline/12_android_server_comparison"


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


def metric_summary(df: pd.DataFrame, pred_col: str, latency_col: str, prefix: str):
    valid = df[df[pred_col].notna() & (df[pred_col].astype(int) >= 0)].copy()
    if valid.empty:
        return {
            f"{prefix}_rows": 0,
            f"{prefix}_accuracy": 0.0,
            f"{prefix}_precision": 0.0,
            f"{prefix}_recall": 0.0,
            f"{prefix}_f1": 0.0,
            f"{prefix}_latency_mean_ms": 0.0,
            f"{prefix}_latency_p95_ms": 0.0,
        }

    y_true = valid["true_label"].astype(int)
    y_pred = valid[pred_col].astype(int)
    latencies = valid[latency_col].astype(float).tolist() if latency_col in valid else []
    if latency_col.endswith("_sec"):
        latencies = [value * 1000 for value in latencies]

    return {
        f"{prefix}_rows": int(len(valid)),
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_latency_mean_ms": float(statistics.mean(latencies)) if latencies else 0.0,
        f"{prefix}_latency_p95_ms": percentile(latencies, 95),
    }


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    index = round((pct / 100) * (len(values) - 1))
    return float(values[index])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--android_csv", default="android_benchmark_results.csv")
    parser.add_argument("--server_csv", default="android_server_baseline_results.csv")
    parser.add_argument("--output_csv", default="android_vs_server_comparison.csv")
    parser.add_argument("--skip_mlflow", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    android = pd.read_csv(args.android_csv)
    server = pd.read_csv(args.server_csv)

    merged = android.merge(
        server,
        on=["sample_id", "true_label"],
        suffixes=("_android", "_server"),
        how="inner",
    )
    if merged.empty:
        raise RuntimeError("No shared sample_id rows between Android and server benchmark files.")

    merged["android_matches_server"] = (
        merged["pipeline_prediction"].fillna(-1).astype(int)
        == merged["predicted_label"].fillna(-1).astype(int)
    )
    merged["android_classifier_matches_server"] = (
        merged["classifier_only_prediction"].fillna(-1).astype(int)
        == merged["predicted_label"].fillna(-1).astype(int)
    )
    merged.to_csv(args.output_csv, index=False)

    summary = {}
    summary.update(
        metric_summary(
            merged,
            pred_col="predicted_label",
            latency_col="total_latency_sec",
            prefix="server_pipeline",
        )
    )
    summary.update(
        metric_summary(
            merged,
            pred_col="pipeline_prediction",
            latency_col="total_latency_ms",
            prefix="android_pipeline",
        )
    )
    summary.update(
        metric_summary(
            merged,
            pred_col="classifier_only_prediction",
            latency_col="classifier_only_latency_ms",
            prefix="android_classifier_only",
        )
    )
    summary["shared_rows"] = int(len(merged))
    summary["pipeline_agreement_rate"] = float(merged["android_matches_server"].mean())
    summary["classifier_only_agreement_rate"] = float(merged["android_classifier_matches_server"].mean())

    print(json.dumps(summary, indent=2))

    if configure_tracking(args.skip_mlflow):
        with mlflow.start_run(run_name="android_vs_server_comparison"):
            mlflow.set_tag("pipeline_stage", "12_android_server_comparison")
            mlflow.log_param("android_csv", args.android_csv)
            mlflow.log_param("server_csv", args.server_csv)
            for key, value in summary.items():
                mlflow.log_metric(key, float(value))
            log_dataframe_artifact(merged, Path(args.output_csv).name, "comparison")
            log_json_artifact(summary, "android_vs_server_summary.json", "comparison")


if __name__ == "__main__":
    main()
