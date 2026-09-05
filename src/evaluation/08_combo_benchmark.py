import os
import time
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import dagshub
import mlflow
from dotenv import load_dotenv
from src.evaluation.inference_pipeline import InferencePipeline
from src.utils.data_access import ensure_audio_holdout
from src.utils.mlflow_reporting import log_benchmark_plots, log_classification_artifacts, log_dataframe_artifact

def get_dir_size(path):
    if not os.path.exists(path):
        if "openai/whisper-tiny" in str(path):
            return 151 * 1024 * 1024 # Approx 151 MB for FP16 Whisper Tiny
        return 0
    if os.path.isfile(path):
        return os.path.getsize(path)
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def evaluate_combinations():
    print("--- Running E2E Combinatorial Benchmarks ---")
    
    audio_dir = "data/large_audio_test"
    ensure_audio_holdout(audio_dir)
        
    files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    files.sort()
    assert len(files) > 0, f"CRITICAL ERROR: No .wav files found in {audio_dir}."
    
    manifest_path = os.path.join(audio_dir, "manifest.csv")
    manifest = pd.read_csv(manifest_path) if os.path.exists(manifest_path) else None
    
    assert manifest is not None, f"CRITICAL ERROR: {manifest_path} not found. Audio benchmark cannot proceed."
    assert len(files) == len(manifest), "CRITICAL ERROR: Downloaded .wav files do not match expected manifest count."
        
    classifier_paths = {
        "gguf_q8": "models/gguf_classifier/classifier_q8_0.gguf",
        "gguf_q4": "models/gguf_classifier/classifier_q4_k_m.gguf",
    }
    whisper_paths = {
        "fp16": "openai/whisper-tiny.en",
        "q8_0": "models/ggml_whisper/whisper_q8_0.bin",
        "q5_1": "models/ggml_whisper/whisper_q5_1.bin",
    }

    # Keep the CPU deployment matrix intentionally small:
    # baseline, likely production candidate, ASR-small candidate, and smallest CPU candidate.
    combinations = [
        ("fp16", "fp16"),
        ("gguf_q8", "q8_0"),
        ("gguf_q8", "q5_1"),
        ("gguf_q4", "q5_1"),
    ]
    
    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    if repo_owner and repo_name:
        dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
        mlflow.set_experiment("scam-detection/refactored_pipeline/08_combo_benchmark")
        
    results = []
    
    # We create a dummy config that we'll inject values into
    with open("configs/inference_config.json", "r") as f:
        base_config = json.load(f)
        
    for clf, asr in combinations:
        print(f"\\n--- Benchmarking Combo: Classifier={clf} | ASR={asr} ---")
        
        run_name = f"combo_{clf}_{asr}"
        config = base_config.copy()
        
        config["classifier_backend"] = "gguf" if clf.startswith("gguf") else "fp16"
        config["asr_backend"] = "fp16" if asr == "fp16" else "gguf"
        
        if clf in classifier_paths:
            config["classifier_model_path"] = classifier_paths[clf]
            config["gguf_classifier_head_path"] = "models/gguf/gguf_classifier_head.joblib"
        else:
            config["fp16_classifier_model_name"] = "./scam-classifier-model-transcript-lora"
            
        if asr == "fp16":
            config["fp16_asr_model_name"] = whisper_paths[asr]
        else:
            config["asr_model_path"] = whisper_paths[asr]
            
        try:
            pipeline = InferencePipeline(config)
        except Exception as e:
            print(f"Failed to initialize pipeline for {clf}+{asr}: {e}")
            continue
            
        y_true = []
        y_pred = []
        latencies = []
        prediction_rows = []
        
        for idx, row in manifest.iterrows():
            audio_path = os.path.join(audio_dir, os.path.basename(row['file']))
            if not os.path.exists(audio_path): continue
            
            y_true.append(row['label'])
            
            start_time = time.time()
            try:
                res = pipeline.process_audio(audio_file=audio_path)
                y_pred.append(res['prediction'])
                prediction_rows.append(
                    {
                        "file": os.path.basename(audio_path),
                        "label": row["label"],
                        "prediction": res["prediction"],
                        "asr_latency_sec": res["metrics"].get("asr_latency"),
                        "classifier_latency_sec": res["metrics"].get("classifier_latency"),
                        "total_latency_sec": res["metrics"].get("total_latency"),
                        "transcript": res.get("transcript", ""),
                    }
                )
            except Exception as e:
                print(f"Failed prediction on {audio_path}: {e}")
                y_pred.append(0)
                prediction_rows.append(
                    {
                        "file": os.path.basename(audio_path),
                        "label": row["label"],
                        "prediction": 0,
                        "error": str(e),
                    }
                )
            end_time = time.time()
            latencies.append(end_time - start_time)

        if not y_true:
            print(f"No valid rows evaluated for {clf}+{asr}. Skipping combo.")
            continue

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        avg_lat = sum(latencies) / len(latencies)
        
        clf_path = config.get("classifier_model_path") if clf != "fp16" else config.get("fp16_classifier_model_name")
        asr_path = config.get("asr_model_path") if asr != "fp16" else config.get("fp16_asr_model_name")
        
        clf_size = get_dir_size(clf_path)
        asr_size = get_dir_size(asr_path)
        total_size_mb = (clf_size + asr_size) / (1024 * 1024)
        
        print(f"Results -> Acc: {acc:.4f}, F1: {f1:.4f}, Latency: {avg_lat:.3f}s, Size: {total_size_mb:.1f}MB")
        
        res_dict = {
            "classifier": clf,
            "asr": asr,
            "accuracy": acc,
            "f1": f1,
            "latency": avg_lat,
            "size_mb": total_size_mb
        }
        results.append(res_dict)
        
        if repo_owner and repo_name:
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("project_stage", "refactored_pipeline")
                mlflow.set_tag("pipeline_stage", "08_combo_benchmark")
                mlflow.log_param("classifier_variant", clf)
                mlflow.log_param("asr_variant", asr)
                mlflow.log_param("classifier_backend", config["classifier_backend"])
                mlflow.log_param("asr_backend", config["asr_backend"])
                mlflow.log_param("classifier_path_or_model", clf_path)
                mlflow.log_param("asr_path_or_model", asr_path)
                mlflow.log_metric("evaluated_rows", len(y_true))
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("latency_sec", avg_lat)
                mlflow.log_metric("size_mb", total_size_mb)
                mlflow.log_artifact(manifest_path, artifact_path="dataset")
                log_dataframe_artifact(pd.DataFrame(prediction_rows), f"{run_name}_predictions.csv", "predictions")
                log_classification_artifacts(y_true, y_pred, artifact_path="evaluation", prefix=run_name)
                
    # Save combo benchmark results
    df_res = pd.DataFrame(results)
    df_res.to_csv("combo_benchmark_results.csv", index=False)
    if repo_owner and repo_name and not df_res.empty:
        with mlflow.start_run(run_name="combo_benchmark_summary"):
            mlflow.set_tag("project_stage", "refactored_pipeline")
            mlflow.set_tag("pipeline_stage", "08_combo_benchmark")
            mlflow.log_metric("num_combinations_evaluated", len(df_res))
            log_dataframe_artifact(df_res, "combo_benchmark_results.csv", "benchmarks")
            plot_df = df_res.copy()
            plot_df["combination"] = plot_df["classifier"].astype(str) + "+" + plot_df["asr"].astype(str)
            log_benchmark_plots(plot_df, "combination", ["accuracy", "f1", "latency", "size_mb"], "benchmarks", "combo")
    print("Combinatorial benchmarking complete. Results saved to combo_benchmark_results.csv")

if __name__ == "__main__":
    evaluate_combinations()
