# Final Project Presentation Outline

Course guideline target: approximately 25 slides, 30 minutes presentation, 10 minutes Q&A.

Presentation style: technical journey for a Practical Machine Learning audience. The deck should not be a business pitch and should not become a metrics dump. Each stage should explain the decision, the evidence, and how it motivated the next stage.

## Narrative Spine

```text
Problem Definition
  -> Data evolution
  -> Model evolution
  -> Audio-domain adaptation
  -> Optimization and quantization
  -> CPU deployment
  -> Reproducibility and MLOps
```

Core journey:

```text
DistilBERT baseline
  -> expanded universal written dataset
  -> ModernBERT long-context classifier
  -> ASR transcript retraining
  -> LoRA/PEFT efficiency
  -> GGUF and calibrated PTQ optimization
  -> sequential audio pipeline benchmark
  -> Hugging Face Spaces deployment
```

## Proposed Slide Plan

1. **Title**
   - Project name, course, author/team.
   - One-line scope: audio-to-scam detection with CPU-oriented deployment.

2. **Problem Definition**
   - Detect scam vs legitimate phone calls from audio.
   - Practical constraints: latency, privacy, reproducibility, deployment cost.

3. **Task Formulation**
   - Input: audio recording.
   - Intermediate: ASR transcript.
   - Output: scam/legitimate prediction.
   - Framing: supervised binary classification with ASR preprocessing.

4. **Literature Survey / Technical Background**
   - Transformer classifiers for text classification.
   - ASR plus downstream NLP pipeline.
   - PEFT/LoRA for efficient adaptation.
   - Post-training quantization and CPU inference.

5. **End-to-End System Architecture**
   - Diagram: raw data -> processed splits -> staged training -> quantization -> inference app.
   - Mention DagsHub/MLflow as the experiment and artifact backbone.

6. **Data Journey Overview**
   - Stage 1: composite written spam data.
   - Stage 1.5: expanded universal written corpus.
   - Stage 2: ASR transcript domain.
   - Frozen holdout and PTQ calibration split.

7. **Data Collection**
   - Kaggle/SMS/Enron/phishing/composite sources.
   - Synthetic scam examples.
   - Teeconnie legitimate call data.
   - ASR transcript data.
   - DagsHub artifact-first reproducibility.

8. **Data Preparation**
   - Cleaning, label normalization, deduplication.
   - Source domain/source dataset columns.
   - Stratified train/validation/test split.
   - PTQ calibration split sampled only from train.
   - Leakage validation script.

9. **Exploratory Data Analysis**
   - Class balance.
   - Source distribution.
   - Text length and tokenizer length.
   - Spoken ASR disfluency/domain shift.
   - Why long-context modeling is justified.

10. **Stage 1: DistilBERT Baseline**
    - Why DistilBERT first: fast, familiar, strong baseline.
    - Trained only on original composite data.
    - Key limitation: 512-token context.

11. **DistilBERT Results**
    - Accuracy/F1/confusion matrix.
    - Use MLflow metric callout.
    - Explain what this establishes and what it cannot solve.

12. **Stage 1.5: Universal Written Dataset**
    - Why data expansion was needed.
    - Added broader written sources and legitimate-call examples.
    - Explain class-balance/data-source decisions.

13. **ModernBERT Architecture Choice**
    - Long context up to 8192 tokens.
    - Better fit for long call transcripts.
    - Contrast with DistilBERT truncation risk.

14. **Full ModernBERT Evidence**
    - Historical full fine-tune metrics from MLflow.
    - Accuracy/F1/precision/recall.
    - Training and inference cost tradeoff.

15. **LoRA ModernBERT Training**
    - Why PEFT: reduce training cost while retaining quality.
    - Trainable parameter ratio around 2.23%.
    - 2x T4 DDP setup.
    - Compare final LoRA metrics against full fine-tune once run completes.

16. **Stage 2: ASR Transcript Retraining**
    - Why written text is not enough.
    - Spoken calls include disfluencies, ASR noise, conversational structure.
    - Retrain/adapt the universal model on transcript-domain data.

17. **Transcript Retraining Results**
    - Validation metrics.
    - Confusion matrix if available.
    - Explain what changed from text-only to audio-transcript domain.

18. **Optimization Motivation**
    - Final deployment target is CPU/local inference.
    - Need latency and model size reduction.
    - ASR dominates runtime, classifier size matters for deployability.

19. **Classifier Quantization**
    - FP16 baseline.
    - GGUF Q8_0.
    - GGUF Q4_K_M.
    - ONNX calibrated INT8 PTQ as an additional side experiment.
    - Clarify GGUF as weight-only post-training quantization.

20. **Whisper Quantization**
    - F16 Whisper baseline.
    - Q8_0.
    - Q4_K.
    - Explain why ASR variants are benchmarked separately.

21. **Sequential Pipeline Benchmarking**
    - Audio -> Whisper -> transcript -> classifier.
    - Essential CPU combinations only:
      - FP16 classifier + FP16 Whisper.
      - GGUF Q8 classifier + Q8 Whisper.
      - GGUF Q8 classifier + Q4 Whisper.
      - GGUF Q4 classifier + Q4 Whisper.
    - Metrics: accuracy/F1, ASR latency, classifier latency, total latency, size.

22. **Best Pipeline Selection**
    - Scoring rule: accuracy, latency, model size.
    - `configs/inference_config.json` stores the winning backend combination.
    - Explain selected tradeoff once final benchmark completes.

23. **Deployment**
    - Hugging Face Spaces.
    - Gradio interface.
    - Config-driven inference pipeline.
    - DagsHub artifact loading.
    - CPU/local privacy angle.

24. **MLOps and Reproducibility**
    - DagsHub MLflow experiments.
    - Artifact storage and retrieval.
    - Dataset validation.
    - Unified branch pipeline.
    - Makefile/run scripts.

25. **Conclusions and Future Directions**
    - Key lessons:
      - data evolution mattered as much as model choice.
      - LoRA can be sufficient if final metrics match full fine-tune.
      - GGUF is practical for CPU deployment.
      - ASR quality/latency drives end-user performance.
    - Future work:
      - larger audio holdout.
      - streaming ASR.
      - compact student-model distillation for future device deployment.
      - privacy-first call recording workflow.

26. **Q&A**
    - Keep as final slide if required, otherwise merge with slide 25.

## Evidence Placement

Use numbers selectively:

- One compact metric table per major model stage.
- One confusion matrix where it is most explanatory.
- One optimization table for latency/size/accuracy tradeoff.
- MLflow run IDs as small footers or callouts, not as the main content.
- DagsHub/MLflow screenshots only when they prove reproducibility.

Avoid:

- listing every run on slides.
- over-documenting one-off failures.
- making optimization look like a disconnected experiment.

## Suggested Diagrams

- End-to-end system architecture.
- Data lineage and split flow.
- Model progression timeline.
- Inference pipeline: audio -> ASR -> classifier -> prediction.
- Optimization comparison chart: quality vs latency/size.

## Slide Status Tags

Use these labels if the deck is drafted before every run finishes:

```text
Implemented
Validated
Running
Next validation step
```

Avoid language like:

```text
missing
failed
manual recovery
not complete
```

The deck should present the intended engineering workflow and the final validated evidence.
