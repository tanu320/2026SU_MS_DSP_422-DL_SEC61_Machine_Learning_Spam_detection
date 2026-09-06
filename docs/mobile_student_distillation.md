# Mobile Student Distillation

The production demo remains the Hugging Face Spaces CPU pipeline using GGUF/GGML artifacts. The mobile-oriented path is now handled through teacher-student distillation rather than exporting a large ModernBERT graph through QAT/ONNX.

## Goal

Train a compact student classifier that mimics the transcript-trained ModernBERT teacher while keeping the original supervised scam/legitimate labels.

Default setup:

- Teacher: transcript-trained ModernBERT from `05_transcript_modernbert`
- Student: `microsoft/MiniLM-L12-H384-uncased`
- Training split: `data/processed/global_train.csv`
- Validation split: `data/processed/global_val.csv`
- Default filter: `source_domain == "spoken_asr"`
- Loss: `alpha * cross_entropy + (1 - alpha) * KL(student_logits, teacher_logits)`

The frozen global test split should remain untouched until final model selection.

## Smoke Test

This validates the plumbing only: DagsHub artifact fetch, teacher-logit generation,
student training loop, evaluation, and MLflow artifact logging. It is expected to
score poorly with only five optimization steps and should not create a registry
model version.

```bash
python src/models/07_distill_mobile_student.py \
  --teacher_run_id 62ffeee7d6d446babb855d5c4af082ce \
  --student_model_name minilm \
  --max_train_samples 128 \
  --max_eval_samples 64 \
  --max_steps 5 \
  --output_dir ./scam-classifier-model-mobile-student-smoke
```

## Full MiniLM Student

```bash
python src/models/07_distill_mobile_student.py \
  --teacher_run_id 62ffeee7d6d446babb855d5c4af082ce \
  --student_model_name minilm \
  --epochs 4 \
  --batch_size 16 \
  --max_length 512 \
  --temperature 2.0 \
  --alpha 0.5 \
  --register_model \
  --output_dir ./scam-classifier-model-mobile-student
```

## Optional MobileBERT Comparison

```bash
python src/models/07_distill_mobile_student.py \
  --teacher_run_id 62ffeee7d6d446babb855d5c4af082ce \
  --student_model_name mobilebert \
  --epochs 4 \
  --batch_size 16 \
  --max_length 512 \
  --temperature 2.0 \
  --alpha 0.5 \
  --register_model \
  --output_dir ./scam-classifier-model-mobilebert-student
```

## Acceptance Criteria

Use the student only if validation metrics remain close to the transcript-trained teacher. Suggested gate:

- F1 within 1-2 percentage points of the teacher on the same validation split
- materially smaller model footprint than ModernBERT
- faster CPU inference in a follow-up benchmark

If the student misses that gate, keep GGUF/GGML as the deployment path and report distillation as future/mobile optimization work.
