"""Prepare a frozen Android evaluation manifest from held-out audio data.

The output manifest is intentionally small and balanced so it can be copied into
an Android app asset bundle and run on-device without retraining or touching
train/validation/PTQ calibration rows.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dagshub
import mlflow
import pandas as pd
from dotenv import load_dotenv

from src.utils.data_access import ensure_audio_holdout
from src.utils.mlflow_reporting import log_dataframe_artifact, log_json_artifact


EXPERIMENT_NAME = "scam-detection/refactored_pipeline/10_android_eval_manifest"


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
        print("[WARN] Missing DagsHub repo env vars; MLflow logging disabled.")
        return False

    dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return True


def normalize_label(value):
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "scam", "spam", "fraud"}:
            return 1
        if lowered in {"0", "legit", "legitimate", "ham", "safe"}:
            return 0
    return int(value)


def build_manifest(audio_dir: Path, rows_per_class: int, seed: int) -> pd.DataFrame:
    source_manifest = pd.read_csv(audio_dir / "manifest.csv")
    if "file" not in source_manifest.columns or "label" not in source_manifest.columns:
        raise ValueError("Audio holdout manifest must contain file and label columns.")

    source_manifest = source_manifest.copy()
    source_manifest["label"] = source_manifest["label"].map(normalize_label)
    source_manifest["audio_filename"] = source_manifest["file"].map(lambda p: os.path.basename(str(p)))
    source_manifest["audio_path"] = source_manifest["audio_filename"].map(
        lambda filename: str(audio_dir / filename)
    )
    source_manifest = source_manifest[source_manifest["audio_path"].map(lambda p: Path(p).exists())]

    sampled = (
        source_manifest.groupby("label", group_keys=False)
        .apply(lambda group: group.sample(n=min(rows_per_class, len(group)), random_state=seed))
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    sampled.insert(0, "sample_id", [f"android_eval_{i:04d}" for i in range(len(sampled))])

    if "reference_transcript" not in sampled.columns:
        transcript_col = next(
            (col for col in ["transcript", "text", "raw_text"] if col in sampled.columns),
            None,
        )
        sampled["reference_transcript"] = sampled[transcript_col] if transcript_col else ""

    columns = [
        "sample_id",
        "audio_filename",
        "audio_path",
        "label",
        "reference_transcript",
    ]
    optional_columns = [c for c in ["duration_sec", "source", "source_domain"] if c in sampled.columns]
    return sampled[columns + optional_columns]


def write_android_bundle(manifest: pd.DataFrame, output_dir: Path):
    audio_out = output_dir / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    android_manifest = manifest.copy()
    android_manifest["audio_asset_path"] = android_manifest["audio_filename"].map(
        lambda filename: f"benchmark/audio/{filename}"
    )

    for _, row in android_manifest.iterrows():
        shutil.copy2(row["audio_path"], audio_out / row["audio_filename"])

    android_manifest.drop(columns=["audio_path"]).to_csv(output_dir / "android_eval_manifest.csv", index=False)
    return android_manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", default="data/large_audio_test")
    parser.add_argument("--rows_per_class", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", default="data/processed/android_eval_manifest.csv")
    parser.add_argument("--android_asset_bundle_dir", default=None)
    parser.add_argument("--skip_mlflow", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    audio_dir = Path(args.audio_dir)
    ensure_audio_holdout(str(audio_dir))

    manifest = build_manifest(audio_dir, args.rows_per_class, args.seed)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_csv, index=False)

    print(f"Wrote frozen Android eval manifest: {output_csv}")
    print(manifest["label"].value_counts().sort_index().to_dict())

    android_manifest = None
    if args.android_asset_bundle_dir:
        android_manifest = write_android_bundle(manifest, Path(args.android_asset_bundle_dir))
        print(f"Wrote Android asset benchmark bundle: {args.android_asset_bundle_dir}")

    if configure_tracking(args.skip_mlflow):
        with mlflow.start_run(run_name="android_eval_manifest"):
            mlflow.set_tag("pipeline_stage", "10_android_eval_manifest")
            mlflow.log_param("source_audio_dir", str(audio_dir))
            mlflow.log_param("rows_per_class", args.rows_per_class)
            mlflow.log_param("seed", args.seed)
            mlflow.log_metric("total_rows", len(manifest))
            for label, count in manifest["label"].value_counts().sort_index().items():
                mlflow.log_metric(f"label_{label}_rows", int(count))
            mlflow.log_artifact(str(output_csv), artifact_path="manifest")
            log_dataframe_artifact(manifest, "android_eval_manifest.csv", "manifest")
            if android_manifest is not None:
                log_dataframe_artifact(android_manifest, "android_asset_manifest.csv", "manifest")
            log_json_artifact(
                {
                    "label_distribution": {
                        str(k): int(v) for k, v in manifest["label"].value_counts().sort_index().items()
                    },
                    "leakage_policy": "sampled only from frozen audio holdout; not train/val/PTQ/QAT",
                },
                "android_eval_manifest_profile.json",
                "manifest",
            )


if __name__ == "__main__":
    main()
