---
title: Scam Detection AI
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.1"
python_version: 3.10.13
app_file: src/deployment/app.py
pinned: false
---

# 🛡️ ScamShield: Edge-Optimized Voice Scam Detection AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![llama.cpp](https://img.shields.io/badge/Quantization-GGUF%20%2F%20llama.cpp-lightgrey)
![MLflow](https://img.shields.io/badge/Tracking-MLflow%20%2F%20DagsHub-blue)

An end-to-end, edge-deployable Machine Learning pipeline designed to detect malicious intent in real-time voice calls. By cascading a **Whisper ASR** model into a fine-tuned **ModernBERT** (8K context window) sequence classifier, this system processes continuous audio streams and flags social engineering attacks with high precision.

---

## 🚀 Key Engineering Highlights

This project was engineered to solve the constraints of deploying state-of-the-art NLP models to edge hardware and shared CPU tiers without sacrificing accuracy.

* **Parameter-Efficient Fine-Tuning (PEFT):** Fine-tuned ModernBERT (149M params) using LoRA (`r=16`), updating only ~2% of parameters. This allowed the model to learn complex manipulative linguistic features on consumer GPUs while avoiding catastrophic forgetting.
* **Long-Context Awareness:** Replaced traditional DistilBERT (512 token limit / ~2.5 mins of audio) with ModernBERT (8,192 token limit / ~41 mins of audio) to ensure scammers cannot bypass detection by burying the payload deep within a long phone call.
* **Edge Quantization (GGUF):** Compressed the FP32 PyTorch pipeline using `llama.cpp` into GGUF format, utilizing Memory Mapping (`mmap`) and CPU vector instructions to achieve **1.98s latency** on standard shared CPUs.
* **Domain Adaptation:** Adapted the model from clean written text (Phase 1) to noisy Whisper ASR transcripts (Phase 2), making the classifier highly robust against phonetic spelling errors and missing punctuation.
* **Strict CI/CD Data Governance:** Engineered an automated data pipeline that deterministically isolates a 20% global holdout set. Implemented strict automated checks that fail the build if exact string overlap is detected between train and test sets, mathematically guaranteeing zero data leakage.

---

## 🏗️ Architecture & Pipeline

The codebase is highly modular and tracked end-to-end via DagsHub MLflow.

### 1. Data Ingestion (`src/data/`)
Downloads raw sources (Kaggle Phishing/Enron/SMS, synthetic LLM JSONs, and raw ASR transcripts). A 20% **Global Hold-Out Set** is carved out and frozen *before* any modeling occurs. 

### 2. Model Training (`src/models/`)
* **Baseline:** DistilBERT trained on the written corpus.
* **Universal:** ModernBERT trained on the expanded written corpus.
* **Domain Adaptation:** ModernBERT fine-tuned exclusively on ASR spoken transcripts to learn noise-resilience.

### 3. Optimization (`src/optimization/`)
Converts the final PyTorch ModernBERT model into CPU-serving essentials: **FP16 baseline**, **GGUF Q8_0**, and **GGUF Q4_K_M** via `llama.cpp`. Also contains experiments for Static Post-Training Quantization (PTQ) via ONNX Runtime using KL-Divergence (Entropy) calibration.

### 4. Evaluation & Benchmarking (`src/evaluation/`)
Runs all optimized E2E pipelines against the frozen Global Hold-Out Set, logging inference latency, model sizing, and F1 scores directly to DagsHub MLflow.

---

## 🛠️ Quickstart & Reproducibility

### Environment Setup
1. Clone the repo and checkout the `refactor/unified-pipeline` branch.
2. Install the locked dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your `.env` variables (DagsHub credentials for MLflow logging & HuggingFace token):
   ```
   DAGSHUB_REPO_OWNER="your-owner"
   DAGSHUB_REPO_NAME="your-repo"
   MLFLOW_TRACKING_USERNAME="..."
   MLFLOW_TRACKING_PASSWORD="..."
   ```

### Run the Pipeline
To execute the canonical journey from raw data download all the way to final benchmarking:

```bash
make all
# OR
bash run_pipeline.sh
```

### Fast LoRA Completion Path
To complete the end-to-end pipeline quickly using LoRA adapters, merge the adapter, and quantize:

```bash
# 1. Train the Universal Model
python src/models/04_train_universal_modernbert.py \
  --finetune_method lora \
  --epochs 4 \
  --output_dir ./scam-classifier-model-universal-lora

# 2. Perform Domain Adaptation on ASR Transcripts
python src/models/05_retrain_transcript_modernbert.py \
  --model_name ./scam-classifier-model-universal-lora \
  --finetune_method lora \
  --epochs 4 \
  --output_dir ./scam-classifier-model-transcript-lora

# 3. Quantize to GGUF for CPU Edge Deployment
python src/optimization/06_ptq_modernbert.py \
  --model_name ./scam-classifier-model-transcript-lora
```

### Live Deployment
Launch the Gradio App and FastAPI inference servers locally:
```bash
python src/deployment/api_server.py
```
