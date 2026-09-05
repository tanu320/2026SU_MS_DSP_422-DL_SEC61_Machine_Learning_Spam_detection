"""
Exports the ASR (Whisper) and Classifier models to GGUF/GGML format
for efficient Hugging Face Spaces/local CPU execution via llama.cpp/whisper.cpp.
Quantizes them to the CPU-serving essentials: F16 baseline, Q8_0, and Q4_K_M/Q4_K.

Also provides an optional calibrated ONNX Runtime static INT8 PTQ path for
GPU/CPU benchmark evidence using data/processed/ptq_calibration.csv.
"""

import os
import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

import dagshub
import mlflow
from dotenv import load_dotenv
from src.utils.data_access import ensure_processed_data
from src.utils.mlflow_reporting import (
    log_benchmark_plots,
    log_classification_artifacts,
    log_dataframe_artifact,
    log_split_profile,
)

STAGE_NAME = "06_ptq_modernbert"

def run_cmd(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)

def upload_to_dagshub(local_path, remote_path, stage):
    if remote_path.startswith("artifacts/") and not remote_path.startswith("artifacts/refactored_pipeline/"):
        remote_path = remote_path.replace("artifacts/", "artifacts/refactored_pipeline/")
        
    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    token = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if repo_owner and repo_name and token:
        try:
            print(f"Uploading {local_path} to DagsHub S3...")
            dagshub.auth.add_app_token(token)
            s3_client = dagshub.get_repo_bucket_client(f"{repo_owner}/{repo_name}")
            s3_client.upload_file(local_path, repo_name, remote_path)
            print(f"  [SUCCESS] Uploaded to S3: {remote_path}")
        except Exception as e:
            print(f"  [FAILED] S3 Upload failed: {e}")

def download_from_dagshub(remote_path, local_path):
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
        print(f"  [SUCCESS] Downloaded S3 artifact: {remote_path} -> {local_path}")
        return True
    except Exception as e:
        print(f"  [WARN] Could not download {remote_path}: {e}")
        return False

def ensure_gguf_classifier_head(local_path="models/gguf/gguf_classifier_head.joblib"):
    if os.path.exists(local_path):
        return local_path

    remote_candidates = [
        local_path,
        f"artifacts/refactored_pipeline/{STAGE_NAME}/gguf/gguf_classifier_head.joblib",
        f"artifacts/{STAGE_NAME}/gguf/gguf_classifier_head.joblib",
        "models/gguf/gguf_classifier_head.joblib",
        "artifacts/feature/phase-2-audio-asr/gguf/gguf_classifier_head.joblib",
        "artifacts/feature/phase-3.5-benchmark/gguf_classifier_head.joblib",
        "artifacts/feature/phase-3.5-benchmark/gguf/gguf_classifier_head.joblib",
    ]
    for remote_path in remote_candidates:
        if download_from_dagshub(remote_path, local_path):
            return local_path

    return None

def mean_pool_llama_embedding(raw_embedding):
    import numpy as np

    arr = np.array(raw_embedding)
    if arr.ndim == 3:
        return np.mean(arr[0], axis=0)
    if arr.ndim == 2:
        return np.mean(arr, axis=0)
    return arr

def train_gguf_classifier_head(
    gguf_model_path,
    train_data="data/processed/global_train.csv",
    train_rows=5000,
    output_path="models/gguf/gguf_classifier_head.joblib",
):
    print("\n--- Training GGUF Classification Head ---")
    if not os.path.exists(gguf_model_path):
        raise FileNotFoundError(f"Cannot train GGUF head; model not found: {gguf_model_path}")

    if not os.path.exists(train_data):
        ensure_processed_data([train_data])

    train_df = pd.read_csv(train_data).dropna(subset=["text", "label"])
    if len(train_df) > train_rows:
        train_df = train_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(
                n=min(len(x), max(1, round(train_rows * len(x) / len(train_df)))),
                random_state=42,
            )
        )
        if len(train_df) > train_rows:
            train_df = train_df.sample(n=train_rows, random_state=42)

    from llama_cpp import Llama
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    n_threads = max(1, (os.cpu_count() or 2) - 1)
    print(
        f"Extracting GGUF embeddings for {len(train_df)} training rows "
        f"using {n_threads} CPU thread(s)...",
        flush=True,
    )
    llm = Llama(
        model_path=gguf_model_path,
        verbose=False,
        embedding=True,
        n_ctx=1024,
        n_batch=512,
        n_threads=n_threads,
    )
    embeddings = []
    labels = []
    embed_start = time.time()
    for idx, (_, row) in enumerate(train_df.iterrows(), start=1):
        embeddings.append(mean_pool_llama_embedding(llm.embed(str(row["text"])[:5000])))
        labels.append(int(row["label"]))
        if idx == 1 or idx % 50 == 0 or idx == len(train_df):
            elapsed = time.time() - embed_start
            rows_per_sec = idx / elapsed if elapsed else 0
            remaining = (len(train_df) - idx) / rows_per_sec if rows_per_sec else 0
            print(
                f"Embedded {idx}/{len(train_df)} rows "
                f"({rows_per_sec:.2f} rows/s, ETA {remaining / 60:.1f} min)",
                flush=True,
            )

    head = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
    )
    print("Fitting MLP Neural Network head...", flush=True)
    head.fit(embeddings, labels)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(head, output_path)
    print(f"[SUCCESS] Trained GGUF classifier head: {output_path}")
    upload_to_dagshub(output_path, f"artifacts/{STAGE_NAME}/gguf/{os.path.basename(output_path)}", STAGE_NAME)
    upload_to_dagshub(output_path, output_path, STAGE_NAME)

    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    if repo_owner and repo_name:
        dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
        mlflow.set_experiment("scam-detection/refactored_pipeline/06_ptq_modernbert")
        with mlflow.start_run(run_name="gguf_classifier_head_training"):
            mlflow.set_tag("project_stage", "refactored_pipeline")
            mlflow.set_tag("pipeline_stage", STAGE_NAME)
            mlflow.log_param("head_model_type", "standard_scaler_mlp_classifier_256_128")
            mlflow.log_param("embedding_model_path", gguf_model_path)
            mlflow.log_param("head_train_data", train_data)
            mlflow.log_metric("head_train_rows", len(train_df))
            mlflow.log_artifact(output_path, artifact_path="models")
            log_split_profile({"gguf_head_train": train_df}, artifact_path="head_training_data")

    return output_path

def export_classifier_to_gguf(model_name="./scam-classifier-model", output_dir="models/gguf_classifier", stage=STAGE_NAME):
    print(f"\n--- Exporting Classifier ({model_name}) to GGUF ---")
    os.makedirs(output_dir, exist_ok=True)
    
    required_models = {
        "classifier_f16.gguf": os.path.join(output_dir, "classifier_f16.gguf"),
        "classifier_q8_0.gguf": os.path.join(output_dir, "classifier_q8_0.gguf"),
        "classifier_q4_k_m.gguf": os.path.join(output_dir, "classifier_q4_k_m.gguf"),
    }
    
    if all(os.path.exists(path) for path in required_models.values()):
        print("Classifier GGUF artifacts already exist locally. Skipping export.")
        return

    remote_prefixes = [
        f"artifacts/refactored_pipeline/{stage}/gguf",
        f"artifacts/{stage}/gguf",
        "models/gguf_classifier",
        "artifacts/feature/phase-2-audio-asr/gguf",
    ]
    fetched_all = True
    for filename, local_path in required_models.items():
        if os.path.exists(local_path):
            continue
        fetched = False
        for prefix in remote_prefixes:
            if download_from_dagshub(f"{prefix}/{filename}", local_path):
                fetched = True
                break
        if not fetched:
            fetched_all = False

    if fetched_all and all(os.path.exists(path) for path in required_models.values()):
        print("Classifier GGUF artifacts resolved from DagsHub.")
        for filename, local_path in required_models.items():
            print(f"[SUCCESS] Available: {local_path} ({os.path.getsize(local_path) / (1024*1024):.2f} MB)")
        return
    
    # Clone llama.cpp if not exists
    if not os.path.exists("llama.cpp"):
        run_cmd(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"])
        # Compile the quantization tool using CMake
        run_cmd(["cmake", "-B", "build"], cwd="llama.cpp")
        run_cmd(["cmake", "--build", "build", "--config", "Release", "-j", "--target", "llama-quantize"], cwd="llama.cpp")

    # Determine if model_name is a local path or HF repo
    if os.path.exists(model_name):
        print(f"Using local model directory: {model_name}")
        local_model_dir = model_name
        
        # FIX: llama.cpp has a bug where it maps both classifier.dense.weight and classifier.out_proj.weight to cls.weight
        # Since llama-cpp-python only uses the embedding for BERT models anyway, we strip the classifier head to prevent collisions.
        import shutil
        from safetensors.torch import load_file, save_file
        
        stripped_dir = local_model_dir + "_stripped"
        if os.path.exists(stripped_dir):
            shutil.rmtree(stripped_dir)
        shutil.copytree(local_model_dir, stripped_dir)
        
        sf_path = os.path.join(stripped_dir, "model.safetensors")
        if os.path.exists(sf_path):
            tensors = load_file(sf_path)
            to_delete = [k for k in tensors.keys() if k.startswith("classifier.")]
            if to_delete:
                for k in to_delete:
                    print(f"Stripping {k} to prevent GGUF collision...")
                    del tensors[k]
                save_file(tensors, sf_path)
        
        local_model_dir = stripped_dir
    else:
        from huggingface_hub import snapshot_download
        print(f"Downloading {model_name} weights locally from HF Hub...")
        local_model_dir = snapshot_download(repo_id=model_name)
    
    # 1. Convert to F16 GGUF
    f16_path = os.path.join(output_dir, "classifier_f16.gguf")
    run_cmd([
        "python", "llama.cpp/convert_hf_to_gguf.py", 
        local_model_dir, 
        "--outfile", f16_path, 
        "--outtype", "f16"
    ])
    
    # 2. Quantize to Q8_0
    q8_path = os.path.join(output_dir, "classifier_q8_0.gguf")
    
    # Locate the compiled llama-quantize binary dynamically
    quantize_bin = None
    for root, dirs, files in os.walk("./llama.cpp/build"):
        if "llama-quantize" in files:
            quantize_bin = os.path.join(root, "llama-quantize")
            break
            
    if not quantize_bin:
        raise FileNotFoundError("Could not find compiled llama-quantize binary in ./llama.cpp/build")
        
    print(f"Quantizing to Q8_0...")
    run_cmd([quantize_bin, f16_path, q8_path, "Q8_0"])
    
    # 3. Quantize to Q4_K_M
    q4_path = os.path.join(output_dir, "classifier_q4_k_m.gguf")
    print(f"Quantizing to Q4_K_M...")
    run_cmd([quantize_bin, f16_path, q4_path, "Q4_K_M"])
    
    # Upload all
    for f in [f16_path, q8_path, q4_path]:
        if os.path.exists(f):
            print(f"[SUCCESS] Generated: {f} ({os.path.getsize(f) / (1024*1024):.2f} MB)")
            upload_to_dagshub(f, f"artifacts/{stage}/gguf/{os.path.basename(f)}", stage)

def export_whisper_to_ggml(model_name="openai/whisper-tiny", output_dir="models/ggml_whisper", stage=STAGE_NAME):
    print(f"\n--- Exporting Whisper ({model_name}) to GGML ---")
    os.makedirs(output_dir, exist_ok=True)

    required_models = {
        "whisper_f16.bin": os.path.join(output_dir, "whisper_f16.bin"),
        "whisper_q8_0.bin": os.path.join(output_dir, "whisper_q8_0.bin"),
        "whisper_q5_1.bin": os.path.join(output_dir, "whisper_q5_1.bin"),
    }
    if all(os.path.exists(path) for path in required_models.values()):
        print("Whisper GGML artifacts already exist locally. Skipping export.")
        return

    remote_prefixes = [
        f"artifacts/refactored_pipeline/{stage}/ggml",
        f"artifacts/{stage}/ggml",
        "models/ggml_whisper",
        "artifacts/feature/phase-2-audio-asr/ggml",
    ]
    fetched_all = True
    for filename, local_path in required_models.items():
        if os.path.exists(local_path):
            continue

        fetched = False
        for prefix in remote_prefixes:
            if download_from_dagshub(f"{prefix}/{filename}", local_path):
                fetched = True
                break

        if not fetched:
            fetched_all = False

    if fetched_all and all(os.path.exists(path) for path in required_models.values()):
        print("Whisper GGML artifacts resolved from DagsHub.")
        for filename, local_path in required_models.items():
            print(f"[SUCCESS] Available: {local_path} ({os.path.getsize(local_path) / (1024*1024):.2f} MB)")
            upload_to_dagshub(local_path, f"artifacts/{stage}/ggml/{filename}", stage)
        return
    
    print("Downloading pre-converted official GGML models directly...")
    import urllib.request
    
    f16_path = os.path.join(output_dir, "whisper_f16.bin")
    q8_path = os.path.join(output_dir, "whisper_q8_0.bin")
    q5_path = os.path.join(output_dir, "whisper_q5_1.bin")
    
    if not os.path.exists(f16_path):
        print("Downloading F16...")
        urllib.request.urlretrieve("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin", f16_path)
        
    if not os.path.exists(q8_path):
        print("Downloading Q8_0...")
        urllib.request.urlretrieve("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny-q8_0.bin", q8_path)
        
    if not os.path.exists(q5_path):
        print("Downloading Q5_1 (Tiny lacks Q4)...")
        urllib.request.urlretrieve("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny-q5_1.bin", q5_path)
        
    # Upload F16
    if os.path.exists(f16_path):
        print(f"[SUCCESS] Generated: {f16_path} ({os.path.getsize(f16_path) / (1024*1024):.2f} MB)")
        upload_to_dagshub(f16_path, f"artifacts/{stage}/ggml/{os.path.basename(f16_path)}", stage)
        
    # Upload Quantized
    for qpath in [q8_path, q5_path]:
        if os.path.exists(qpath):
            print(f"[SUCCESS] Generated: {qpath} ({os.path.getsize(qpath) / (1024*1024):.2f} MB)")
            upload_to_dagshub(qpath, f"artifacts/{stage}/ggml/{os.path.basename(qpath)}", stage)

def evaluate_ptq_degradation(
    stage,
    eval_data="data/processed/global_test.csv",
    eval_rows=512,
    head_train_data="data/processed/global_train.csv",
    head_train_rows=512,
):
    print("\n--- Evaluating PTQ Degradation ---")
    
    try:
        from llama_cpp import Llama
    except ImportError:
        print("llama-cpp-python not installed. Skipping PTQ eval.")
        return
        
    if not os.path.exists(eval_data):
        ensure_processed_data([eval_data])
    eval_source_df = pd.read_csv(eval_data).dropna(subset=["text", "label"])
    if len(eval_source_df) > eval_rows:
        df = eval_source_df.groupby(["label"], group_keys=False).apply(
            lambda x: x.sample(
                n=min(len(x), max(1, round(eval_rows * len(x) / len(eval_source_df)))),
                random_state=42,
            )
        )
        if len(df) > eval_rows:
            df = df.sample(n=eval_rows, random_state=42)
    else:
        df = eval_source_df
    
    models = {
        "F16": f"models/gguf_classifier/classifier_f16.gguf",
        "Q8_0": f"models/gguf_classifier/classifier_q8_0.gguf",
        "Q4_K_M": f"models/gguf_classifier/classifier_q4_k_m.gguf",
    }
    
    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    if repo_owner and repo_name:
        dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
        mlflow.set_experiment("scam-detection/refactored_pipeline/06_ptq_modernbert")
        
    head_path = ensure_gguf_classifier_head()
    if head_path is None:
        head_path = train_gguf_classifier_head(
            gguf_model_path=models["F16"],
            train_data=head_train_data,
            train_rows=head_train_rows,
        )

    import joblib
    gguf_head = joblib.load(head_path)
    results = []

    for name, path in models.items():
        if not os.path.exists(path):
            continue
            
        print(f"Evaluating {name}...")
        llm = Llama(model_path=path, verbose=False, embedding=True)
        
        start = time.time()
        y_true = []
        y_pred = []
        
        import numpy as np
        for idx, row in df.iterrows():
            y_true.append(row['label'])
            
            raw_emb = llm.embed(row['text'][:5000])
            embeds = mean_pool_llama_embedding(raw_emb)
                
            pred_idx = gguf_head.predict([embeds])[0]
            y_pred.append(pred_idx)
                
        end = time.time()
        
        latency = (end - start) / len(df)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"{name}: {latency:.3f} s/req, {size_mb:.2f} MB, Acc: {acc:.4f}, F1: {f1:.4f}")
        results.append(
            {
                "variant": name,
                "latency_sec": latency,
                "size_mb": size_mb,
                "accuracy": acc,
                "f1_score": f1,
                "model_path": path,
                "post_quant_eval_rows": len(df),
                "post_quant_eval_data": eval_data,
            }
        )
        
        if repo_owner and repo_name:
            with mlflow.start_run(run_name=f"ptq_{name}"):
                mlflow.set_tag("project_stage", "refactored_pipeline")
                mlflow.set_tag("pipeline_stage", stage)
                mlflow.log_param("precision_variant", name)
                mlflow.log_param("model_path", path)
                mlflow.log_param("classifier_head_path", head_path)
                mlflow.log_param("quantization_method", "weight_only_gguf")
                mlflow.log_param("post_quant_evaluation_dataset", eval_data)
                mlflow.log_metric("post_quant_eval_rows", len(df))
                mlflow.log_metric("latency_sec", latency)
                mlflow.log_metric("size_mb", size_mb)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)
                log_split_profile({"gguf_post_quant_eval": df}, artifact_path="post_quant_eval")
                log_classification_artifacts(y_true, y_pred, artifact_path="evaluation", prefix=f"ptq_{name.lower()}")
                mlflow.log_artifact(path, artifact_path="models")

    if repo_owner and repo_name and results:
        with mlflow.start_run(run_name="ptq_summary"):
            mlflow.set_tag("project_stage", "refactored_pipeline")
            mlflow.set_tag("pipeline_stage", stage)
            results_df = pd.DataFrame(results)
            log_dataframe_artifact(results_df, "ptq_benchmark_summary.csv", "benchmarks")
            log_benchmark_plots(results_df, "variant", ["accuracy", "f1_score", "latency_sec", "size_mb"], "benchmarks", "ptq")

class TextCalibrationDataReader:
    def __init__(self, df, tokenizer, input_names, max_length=512, batch_size=8):
        self.batches = []
        texts = df["text"].astype(str).tolist()
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            )
            batch = {}
            for name in input_names:
                if name in encoded:
                    batch[name] = encoded[name]
                elif name == "token_type_ids":
                    batch[name] = encoded["input_ids"] * 0
            self.batches.append(batch)
        self.index = 0

    def get_next(self):
        if self.index >= len(self.batches):
            return None
        batch = self.batches[self.index]
        self.index += 1
        return batch

    def rewind(self):
        self.index = 0

def _init_mlflow_experiment():
    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    if repo_owner and repo_name:
        dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
        mlflow.set_experiment("scam-detection/refactored_pipeline/06_ptq_modernbert")
    return repo_owner, repo_name

def _load_classifier_components(model_name, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if model_name.startswith("models:/"):
        components = mlflow.transformers.load_model(model_name, return_type="components")
        tokenizer = components["tokenizer"]
        model = components["model"]
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model

def _onnx_inputs(tokenizer, texts, input_names, max_length):
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    inputs = {}
    for name in input_names:
        if name in encoded:
            inputs[name] = encoded[name]
        elif name == "token_type_ids":
            inputs[name] = encoded["input_ids"] * 0
    return inputs

def _benchmark_onnx(onnx_path, eval_df, tokenizer, max_length, batch_size=8, providers=None):
    import numpy as np
    import onnxruntime as ort
    from sklearn.metrics import accuracy_score, f1_score

    session = ort.InferenceSession(
        onnx_path,
        providers=providers or ["CPUExecutionProvider"],
    )
    input_names = [inp.name for inp in session.get_inputs()]

    y_true = eval_df["label"].astype(int).tolist()
    y_pred = []
    start_time = time.time()
    for start in range(0, len(eval_df), batch_size):
        batch = eval_df.iloc[start:start + batch_size]
        inputs = _onnx_inputs(tokenizer, batch["text"].astype(str).tolist(), input_names, max_length)
        logits = session.run(None, inputs)[0]
        y_pred.extend(np.argmax(logits, axis=-1).astype(int).tolist())
    latency = (time.time() - start_time) / len(eval_df)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "latency_sec": latency,
        "y_true": y_true,
        "y_pred": y_pred,
    }

def run_calibrated_onnx_ptq(
    model_name,
    calibration_data="data/processed/ptq_calibration.csv",
    eval_data="data/processed/global_test.csv",
    output_dir="models/onnx_modernbert",
    max_length=512,
    batch_size=8,
    opset=17,
    eval_rows=512,
):
    print("\n--- Running calibrated ONNX Runtime static INT8 PTQ ---")

    try:
        import numpy as np
        import torch
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as exc:
        print(f"[WARNING] ONNX Runtime PTQ dependencies not installed. Skipping calibrated PTQ: {exc}")
        return

    ensure_processed_data([calibration_data, eval_data])

    os.makedirs(output_dir, exist_ok=True)
    onnx_fp32_path = os.path.join(output_dir, "modernbert_fp32.onnx")
    onnx_int8_path = os.path.join(output_dir, "modernbert_int8_static.onnx")

    calibration_df = pd.read_csv(calibration_data).dropna(subset=["text", "label"])
    eval_df = pd.read_csv(eval_data).dropna(subset=["text", "label"])
    eval_df = eval_df[eval_df["source_domain"] == "spoken_asr"] if "source_domain" in eval_df.columns else eval_df
    if eval_df.empty:
        eval_df = pd.read_csv(eval_data).dropna(subset=["text", "label"])
    if len(eval_df) > eval_rows:
        eval_df = eval_df.groupby(["label"], group_keys=False).apply(
            lambda x: x.sample(
                n=min(len(x), max(1, round(eval_rows * len(x) / len(eval_df)))),
                random_state=42,
            )
        )
        if len(eval_df) > eval_rows:
            eval_df = eval_df.sample(n=eval_rows, random_state=42)

    calibration_texts = set(calibration_df["text"].astype(str))
    eval_texts = set(eval_df["text"].astype(str))
    assert calibration_texts.isdisjoint(eval_texts), "ONNX PTQ calibration data overlaps evaluation data"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = _load_classifier_components(model_name, device=device)

    sample = tokenizer(
        ["representative calibration export sample"],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    sample = {k: v.to(device) for k, v in sample.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
    input_names = list(sample.keys())
    dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
    dynamic_axes["logits"] = {0: "batch"}

    class OnnxSequenceClassifier(torch.nn.Module):
        def __init__(self, base_model, ordered_input_names):
            super().__init__()
            self.base_model = base_model
            self.ordered_input_names = ordered_input_names

        def forward(self, *args):
            inputs = dict(zip(self.ordered_input_names, args))
            return self.base_model(**inputs).logits

    export_model = OnnxSequenceClassifier(model, input_names).to(device).eval()

    if not os.path.exists(onnx_fp32_path):
        print(f"Exporting FP32 ONNX model to {onnx_fp32_path}...")
        with torch.no_grad():
            torch.onnx.export(
                export_model,
                tuple(sample.values()),
                onnx_fp32_path,
                input_names=input_names,
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
                opset_version=opset,
            )
    else:
        print(f"Using existing FP32 ONNX model at {onnx_fp32_path}")

    reader = TextCalibrationDataReader(
        calibration_df,
        tokenizer,
        input_names=input_names,
        max_length=max_length,
        batch_size=batch_size,
    )
    print(f"Quantizing ONNX model with {len(calibration_df)} calibration rows...")
    quantize_static(
        model_input=onnx_fp32_path,
        model_output=onnx_int8_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
    )

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    fp32_metrics = _benchmark_onnx(onnx_fp32_path, eval_df, tokenizer, max_length, batch_size, providers)
    int8_metrics = _benchmark_onnx(onnx_int8_path, eval_df, tokenizer, max_length, batch_size, providers)

    fp32_size_mb = os.path.getsize(onnx_fp32_path) / (1024 * 1024)
    int8_size_mb = os.path.getsize(onnx_int8_path) / (1024 * 1024)
    results_df = pd.DataFrame(
        [
            {
                "variant": "onnx_fp32",
                "accuracy": fp32_metrics["accuracy"],
                "f1_score": fp32_metrics["f1_score"],
                "latency_sec": fp32_metrics["latency_sec"],
                "size_mb": fp32_size_mb,
                "model_path": onnx_fp32_path,
            },
            {
                "variant": "onnx_int8_static",
                "accuracy": int8_metrics["accuracy"],
                "f1_score": int8_metrics["f1_score"],
                "latency_sec": int8_metrics["latency_sec"],
                "size_mb": int8_size_mb,
                "model_path": onnx_int8_path,
            },
        ]
    )

    repo_owner, repo_name = _init_mlflow_experiment()
    if repo_owner and repo_name:
        with mlflow.start_run(run_name="onnx_static_int8_ptq"):
            mlflow.set_tag("project_stage", "refactored_pipeline")
            mlflow.set_tag("pipeline_stage", STAGE_NAME)
            mlflow.set_tag("quantization_family", "calibrated_static_int8")
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("calibration_dataset", calibration_data)
            mlflow.log_param("evaluation_dataset", eval_data)
            mlflow.log_param("calibration_rows", len(calibration_df))
            mlflow.log_param("evaluation_rows", len(eval_df))
            mlflow.log_param("onnx_opset", opset)
            mlflow.log_param("onnx_max_length", max_length)
            mlflow.log_param("calibration_method", "MinMax")
            mlflow.log_param("quant_format", "QDQ")
            mlflow.log_metric("onnx_fp32_accuracy", fp32_metrics["accuracy"])
            mlflow.log_metric("onnx_fp32_f1_score", fp32_metrics["f1_score"])
            mlflow.log_metric("onnx_fp32_latency_sec", fp32_metrics["latency_sec"])
            mlflow.log_metric("onnx_fp32_size_mb", fp32_size_mb)
            mlflow.log_metric("onnx_int8_static_accuracy", int8_metrics["accuracy"])
            mlflow.log_metric("onnx_int8_static_f1_score", int8_metrics["f1_score"])
            mlflow.log_metric("onnx_int8_static_latency_sec", int8_metrics["latency_sec"])
            mlflow.log_metric("onnx_int8_static_size_mb", int8_size_mb)
            mlflow.log_metric("accuracy_delta_int8_minus_fp32", int8_metrics["accuracy"] - fp32_metrics["accuracy"])
            mlflow.log_metric("f1_delta_int8_minus_fp32", int8_metrics["f1_score"] - fp32_metrics["f1_score"])
            mlflow.log_metric("size_reduction_mb", fp32_size_mb - int8_size_mb)
            log_split_profile(
                {
                    "onnx_ptq_calibration": calibration_df,
                    "onnx_ptq_evaluation": eval_df,
                },
                artifact_path="onnx_ptq_dataset_profile",
            )
            log_dataframe_artifact(results_df, "onnx_static_int8_ptq_summary.csv", "onnx_ptq")
            log_benchmark_plots(
                results_df,
                "variant",
                ["accuracy", "f1_score", "latency_sec", "size_mb"],
                "onnx_ptq",
                "onnx_static_int8",
            )
            log_classification_artifacts(
                fp32_metrics["y_true"],
                fp32_metrics["y_pred"],
                artifact_path="onnx_ptq/evaluation",
                prefix="onnx_fp32",
            )
            log_classification_artifacts(
                int8_metrics["y_true"],
                int8_metrics["y_pred"],
                artifact_path="onnx_ptq/evaluation",
                prefix="onnx_int8_static",
            )
            mlflow.log_artifact(calibration_data, artifact_path="onnx_ptq/dataset")
            mlflow.log_artifact(eval_data, artifact_path="onnx_ptq/dataset")
            mlflow.log_artifact(onnx_fp32_path, artifact_path="onnx_ptq/models")
            mlflow.log_artifact(onnx_int8_path, artifact_path="onnx_ptq/models")

    upload_to_dagshub(onnx_fp32_path, f"artifacts/{STAGE_NAME}/onnx/{os.path.basename(onnx_fp32_path)}", STAGE_NAME)
    upload_to_dagshub(onnx_int8_path, f"artifacts/{STAGE_NAME}/onnx/{os.path.basename(onnx_int8_path)}", STAGE_NAME)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="./scam-classifier-model-transcript-lora")
    parser.add_argument("--whisper_name", type=str, default="openai/whisper-tiny")
    parser.add_argument("--skip_onnx_ptq", action="store_true", help="Skip calibrated ONNX Runtime static INT8 PTQ")
    parser.add_argument("--ptq_calibration_data", type=str, default="data/processed/ptq_calibration.csv")
    parser.add_argument("--gguf_eval_data", type=str, default="data/processed/global_test.csv")
    parser.add_argument("--gguf_eval_rows", type=int, default=512)
    parser.add_argument("--gguf_head_train_data", type=str, default="data/processed/global_train.csv")
    parser.add_argument("--gguf_head_train_rows", type=int, default=5000)
    parser.add_argument("--onnx_eval_data", type=str, default="data/processed/global_test.csv")
    parser.add_argument("--onnx_output_dir", type=str, default="models/onnx_modernbert")
    parser.add_argument("--onnx_max_length", type=int, default=512)
    parser.add_argument("--onnx_batch_size", type=int, default=8)
    parser.add_argument("--onnx_opset", type=int, default=17)
    parser.add_argument("--onnx_eval_rows", type=int, default=512)
    args = parser.parse_args()
    stage = STAGE_NAME

    export_classifier_to_gguf(
        model_name=args.model_name,
        output_dir="models/gguf_classifier",
        stage=stage,
    )
    
    export_whisper_to_ggml(
        model_name=args.whisper_name,
        output_dir="models/ggml_whisper",
        stage=stage,
    )

    evaluate_ptq_degradation(
        stage,
        eval_data=args.gguf_eval_data,
        eval_rows=args.gguf_eval_rows,
        head_train_data=args.gguf_head_train_data,
        head_train_rows=args.gguf_head_train_rows,
    )
    if not args.skip_onnx_ptq:
        run_calibrated_onnx_ptq(
            model_name=args.model_name,
            calibration_data=args.ptq_calibration_data,
            eval_data=args.onnx_eval_data,
            output_dir=args.onnx_output_dir,
            max_length=args.onnx_max_length,
            batch_size=args.onnx_batch_size,
            opset=args.onnx_opset,
            eval_rows=args.onnx_eval_rows,
        )
