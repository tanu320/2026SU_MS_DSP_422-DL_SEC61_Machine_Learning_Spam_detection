# Android Validation Runbook

This runbook validates the Android path against the same frozen held-out samples used by
the server/local benchmark.

## 1. Prepare Frozen Android Eval Manifest

Run on the Kaggle/GPU workspace after pulling the latest branch:

```bash
python src/evaluation/10_prepare_android_eval_manifest.py \
  --rows_per_class 25 \
  --android_asset_bundle_dir /path/to/scam-detection-android/app/src/main/assets/benchmark
```

Outputs:

- `data/processed/android_eval_manifest.csv`
- Android asset bundle under `app/src/main/assets/benchmark/`

The manifest is sampled only from the frozen audio holdout, not training, validation, PTQ,
or QAT calibration data.

## 2. Export Android Classifier Asset

After the transcript-adapted classifier exists locally:

```bash
python src/optimization/07_qat_modernbert.py \
  --model_dir ./scam-classifier-model-transcript-lora \
  --output_dir /path/to/scam-detection-android/app/src/main/assets/models
```

This writes:

- `modernbert_qat_int8.onnx`
- tokenizer files
- `model_manifest.json`

The Android classifier contract is:

```text
input_ids + attention_mask -> logits[legitimate, scam]
```

## 3. Run Server Baseline On Same Manifest

```bash
python src/evaluation/11_benchmark_server_baseline.py \
  --manifest data/processed/android_eval_manifest.csv \
  --config configs/best_inference_config.json \
  --output_csv android_server_baseline_results.csv
```

This logs accuracy, F1, latency, and prediction rows to DagsHub MLflow.

## 4. Run Android Benchmark

Open the Android project in Android Studio, build/install on a physical device, then tap:

```text
VALIDATE MODEL ASSETS
RUN FROZEN BENCHMARK
```

The app writes:

```text
/data/data/com.example.scamdetection/files/benchmarks/android_benchmark_results.csv
/data/data/com.example.scamdetection/files/benchmarks/android_metrics_summary.json
```

Pull with adb:

```bash
adb shell run-as com.example.scamdetection \
  cat files/benchmarks/android_benchmark_results.csv > android_benchmark_results.csv

adb shell run-as com.example.scamdetection \
  cat files/benchmarks/android_metrics_summary.json > android_metrics_summary.json
```

## 5. Compare Android Against Server Baseline

```bash
python src/evaluation/12_compare_android_server_results.py \
  --android_csv android_benchmark_results.csv \
  --server_csv android_server_baseline_results.csv \
  --output_csv android_vs_server_comparison.csv
```

Report:

- classifier-only Android accuracy/F1
- full Android audio pipeline accuracy/F1
- server baseline accuracy/F1 on the same sample IDs
- p50/p95 latency
- time-to-actionable-insight
- Android/server prediction agreement

## Current ASR Note

The Android app currently validates Whisper ONNX assets and fails closed if they are missing or
malformed. A fully local Play Store path still needs a verified Android ASR implementation,
preferably whisper.cpp through Android/JNI or a complete Whisper ONNX graph with tokenizer/decoder.
