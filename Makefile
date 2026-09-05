PYTHON ?= python
CUDA_VISIBLE_DEVICES ?= 0,1
NPROC_PER_NODE ?= 2
EPOCHS ?= 4
MAX_LENGTH ?= 2048
MODERNBERT_OUTPUT_DIR ?= ./scam-classifier-model-universal-lora
TRANSCRIPT_OUTPUT_DIR ?= ./scam-classifier-model-transcript-lora
MODERNBERT_MODEL_NAME ?= answerdotai/ModernBERT-base
TEACHER_RUN_ID ?=
TEACHER_RUN_ARG = $(if $(TEACHER_RUN_ID),--teacher_run_id $(TEACHER_RUN_ID),)

.PHONY: \
	download-data download-data-no-mlflow \
	build-data build-data-no-mlflow \
	validate-data data data-no-mlflow \
	train-distilbert train-modernbert train-modernbert-lora train-modernbert-full \
	train-transcript train-transcript-lora quantize evaluate distill-mobile-student all

download-data:
	$(PYTHON) src/data/00_download_raw_data.py

download-data-no-mlflow:
	$(PYTHON) src/data/00_download_raw_data.py --skip_mlflow

build-data: download-data
	$(PYTHON) src/data/02_build_datasets.py

build-data-no-mlflow: download-data-no-mlflow
	$(PYTHON) src/data/02_build_datasets.py --skip_mlflow

validate-data:
	$(PYTHON) src/data/03_validate_partitions.py

data: build-data validate-data

data-no-mlflow: build-data-no-mlflow validate-data

train-distilbert: data
	$(PYTHON) src/models/03_train_baseline_distilbert.py

train-modernbert: data
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) torchrun --nproc_per_node=$(NPROC_PER_NODE) src/models/04_train_universal_modernbert.py \
		--finetune_method lora \
		--epochs $(EPOCHS) \
		--max_length $(MAX_LENGTH) \
		--output_dir $(MODERNBERT_OUTPUT_DIR)

train-modernbert-lora: train-modernbert

train-modernbert-full: data
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) torchrun --nproc_per_node=$(NPROC_PER_NODE) src/models/04_train_universal_modernbert.py \
		--finetune_method full \
		--epochs $(EPOCHS) \
		--max_length $(MAX_LENGTH) \
		--output_dir ./scam-classifier-model-universal-full

train-transcript: train-modernbert
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) torchrun --nproc_per_node=$(NPROC_PER_NODE) src/models/05_retrain_transcript_modernbert.py \
		--model_name $(MODERNBERT_OUTPUT_DIR) \
		--finetune_method lora \
		--epochs $(EPOCHS) \
		--max_length $(MAX_LENGTH) \
		--output_dir $(TRANSCRIPT_OUTPUT_DIR)

train-transcript-lora: train-transcript

quantize: train-transcript
	$(PYTHON) src/optimization/06_ptq_modernbert.py

evaluate: quantize
	$(PYTHON) src/evaluation/07_whisper_quant_benchmark.py
	$(PYTHON) src/evaluation/08_combo_benchmark.py
	$(PYTHON) src/evaluation/09_best_pipeline_selection.py

distill-mobile-student:
	$(PYTHON) src/models/07_distill_mobile_student.py \
		--teacher_model_dir $(TRANSCRIPT_OUTPUT_DIR) \
		$(TEACHER_RUN_ARG) \
		--student_model_name minilm \
		--output_dir ./scam-classifier-model-mobile-student

all:
	bash run_pipeline.sh
