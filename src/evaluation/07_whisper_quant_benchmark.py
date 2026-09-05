import os
import time
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import dagshub
import mlflow
from dotenv import load_dotenv
from src.utils.data_access import ensure_audio_holdout
from src.utils.mlflow_reporting import log_benchmark_plots, log_dataframe_artifact

def _download_from_dagshub(remote_path, local_path):
    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    token = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if not (repo_owner and repo_name and token):
        return False

    try:
        dagshub.auth.add_app_token(token)
        s3_client = dagshub.get_repo_bucket_client(f"{repo_owner}/{repo_name}")
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(repo_name, remote_path, local_path)
        print(f"Downloaded {remote_path} -> {local_path}")
        return True
    except Exception as exc:
        print(f"[WARN] Could not download {remote_path}: {exc}")
        return False

def ensure_whisper_model(path):
    if os.path.exists(path):
        return True

    filename = os.path.basename(path)
    remote_prefixes = [
        "artifacts/refactored_pipeline/06_ptq_modernbert/ggml",
        "artifacts/06_ptq_modernbert/ggml",
        "models/ggml_whisper",
        "artifacts/feature/phase-2-audio-asr/ggml",
    ]
    for prefix in remote_prefixes:
        if _download_from_dagshub(f"{prefix}/{filename}", path):
            return True

    return False

def get_whisper_model_path(variant):
    # Depending on what we exported in 06, we might have these variants
    paths = {
        "F16": "models/ggml_whisper/whisper_f16.bin",
        "Q8_0": "models/ggml_whisper/whisper_q8_0.bin",
        "Q5_1": "models/ggml_whisper/whisper_q5_1.bin"
    }
    return paths.get(variant)

def _segment_text(segments):
    texts = []
    for segment in segments:
        text = getattr(segment, "text", None)
        if text is None and isinstance(segment, dict):
            text = segment.get("text")
        if text:
            texts.append(str(text).strip())
    return " ".join(texts).strip()

def _word_error_rate(reference, hypothesis):
    ref = re.findall(r"\w+", str(reference).lower())
    hyp = re.findall(r"\w+", str(hypothesis).lower())
    if not ref:
        return None
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1] / len(ref)

def evaluate_whisper():
    print("--- Evaluating Whisper Quantization Variants ---")
    
    manifest_path = "data/large_audio_test/manifest.csv"
    
    ensure_audio_holdout()
    
    if not os.path.exists(manifest_path):
        print(f"Manifest not found at {manifest_path}. Skipping.")
        return
        
    try:
        from pywhispercpp.model import Model
    except ImportError:
        print("pywhispercpp not installed. Skipping Whisper eval.")
        return
        
    df = pd.read_csv(manifest_path)
    
    # CPU-serving essentials: FP16 baseline, Q8 quality candidate, and smaller Q5 candidate.
    variants = ["F16", "Q8_0", "Q5_1"]
    
    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    if repo_owner and repo_name:
        dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
        mlflow.set_experiment("scam-detection/refactored_pipeline/07_whisper_quant_benchmark")

    reference_col = next((col for col in ["reference", "reference_text", "transcript", "gold_transcript"] if col in df.columns), None)
    results = []

    for variant in variants:
        path = get_whisper_model_path(variant)
        if not path or not ensure_whisper_model(path):
            print(f"Variant {variant} not found at {path}. Skipping.")
            continue
            
        print(f"Evaluating {variant}...")
        model = Model(path, n_threads=4, print_realtime=False, print_progress=False)
        
        start = time.time()
        transcripts = []
        wers = []
        for _, row in df.iterrows():
            audio_path = f"data/large_audio_test/{os.path.basename(row['file'])}"
            if os.path.exists(audio_path):
                try:
                    segments = model.transcribe(audio_path, new_segment_callback=None)
                    transcript = _segment_text(segments)
                    transcripts.append(
                        {
                            "file": os.path.basename(row["file"]),
                            "variant": variant,
                            "transcript": transcript,
                        }
                    )
                    if reference_col:
                        wer = _word_error_rate(row[reference_col], transcript)
                        if wer is not None:
                            wers.append(wer)
                except Exception as e:
                    print(f"Failed to transcribe {audio_path}: {e}")
        end = time.time()
        
        latency = (end - start) / len(df)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        mean_wer = sum(wers) / len(wers) if wers else None
        print(f"{variant}: {latency:.3f} s/req, {size_mb:.2f} MB")
        results.append(
            {
                "variant": variant,
                "latency_sec": latency,
                "size_mb": size_mb,
                "mean_wer": mean_wer,
                "evaluated_rows": len(df),
                "model_path": path,
            }
        )
        
        if repo_owner and repo_name:
            with mlflow.start_run(run_name=f"whisper_{variant}"):
                mlflow.set_tag("project_stage", "refactored_pipeline")
                mlflow.set_tag("pipeline_stage", "07_whisper_quant_benchmark")
                mlflow.log_param("asr_variant", variant)
                mlflow.log_param("model_path", path)
                mlflow.log_param("manifest_path", manifest_path)
                mlflow.log_metric("evaluated_rows", len(df))
                mlflow.log_metric("latency_sec", latency)
                mlflow.log_metric("size_mb", size_mb)
                if mean_wer is not None:
                    mlflow.log_metric("mean_wer", mean_wer)
                mlflow.log_artifact(manifest_path, artifact_path="dataset")
                if transcripts:
                    log_dataframe_artifact(pd.DataFrame(transcripts), f"whisper_{variant.lower()}_transcripts.csv", "predictions")
                mlflow.log_artifact(path, artifact_path="models")

    if repo_owner and repo_name and results:
        with mlflow.start_run(run_name="whisper_quant_summary"):
            mlflow.set_tag("project_stage", "refactored_pipeline")
            mlflow.set_tag("pipeline_stage", "07_whisper_quant_benchmark")
            results_df = pd.DataFrame(results)
            log_dataframe_artifact(results_df, "whisper_quant_benchmark_summary.csv", "benchmarks")
            log_benchmark_plots(results_df, "variant", ["latency_sec", "size_mb", "mean_wer"], "benchmarks", "whisper")

if __name__ == "__main__":
    evaluate_whisper()
