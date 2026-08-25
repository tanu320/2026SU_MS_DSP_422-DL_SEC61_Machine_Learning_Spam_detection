import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.quantization
import mlflow
import os

def fetch_model_from_dagshub(run_id: str, download_dir: str):
    """
    Downloads the Phase 2 (Transcript Retrained) model from DagsHub MLflow root.
    """
    print(f"Fetching Phase 2 model from MLflow (Run ID: {run_id})...")
    
    # Load environment variables from .env if present
    from dotenv import load_dotenv
    load_dotenv()
    
    # Ensure credentials are set in the environment or fallback to your specific repo
    default_uri = "https://dagshub.com/kureeltanishq/2026SU_MS_DSP_422-DL_SEC61_Machine_Learning_Spam_detection.mlflow"
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", default_uri))
    
    client = mlflow.tracking.MlflowClient()
    
    # Ensure the download directory exists before downloading
    os.makedirs(download_dir, exist_ok=True)
    
    print("Listing artifacts in the root of the MLflow run...")
    artifacts = client.list_artifacts(run_id, "")
    
    if not artifacts:
        print("[ERROR] No artifacts found in this run! Please verify the Run ID.")
        exit(1)
        
    for artifact in artifacts:
        print(f"  Downloading {artifact.path}...")
        client.download_artifacts(run_id, artifact.path, download_dir)
        
    print(f"Artifacts successfully downloaded to: {download_dir}")
    
    # MLflow often nests the model inside subdirectories (e.g. 05_transcript_modernbert/scam-classifier...)
    # We must dynamically search the downloads folder for the exact location of config.json
    model_path = None
    for root, dirs, files in os.walk(download_dir):
        if "config.json" in files and "model.safetensors" in files:
            model_path = root
            break
            
    if model_path is None:
        print("[ERROR] Could not find a valid HuggingFace model (config.json) in the downloaded artifacts!")
        exit(1)
        
    print(f"Found HuggingFace Model Weights at: {model_path}")
    return model_path

def configure_qat(model_path: str):
    """
    Sets up the ModernBERT model for Quantization-Aware Training (QAT).
    """
    print(f"Loading Phase 2 Model for QAT: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Set model to training mode (required for QAT)
    model.train()
    
    # 1. Define the quantization backend (qnnpack is standard for Android ARM)
    model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    
    # [BUG FIX]: PyTorch requires Embedding layers to use a specific float_qparams config
    # because they are lookup tables, not mathematical matrix multiplications.
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            module.qconfig = torch.quantization.float_qparams_weight_only_qconfig
            
    print("Injecting FakeQuantize nodes into PyTorch graph for QAT...")
    torch.quantization.prepare_qat(model, inplace=True)
    
    return model

def export_qat_to_android(qat_model, output_onnx_path="modernbert_qat_int8.onnx"):
    """
    Converts the fake-quantized model to true INT8 and exports to ONNX.
    """
    print("Converting FakeQuantize model to pure INT8...")
    qat_model.eval()
    quantized_model = torch.quantization.convert(qat_model, inplace=False)
    
    print(f"Exporting INT8 model to {output_onnx_path}...")
    dummy_input = {
        "input_ids": torch.randint(0, 30000, (1, 128)),
        "attention_mask": torch.ones(1, 128)
    }
    
    torch.onnx.export(
        quantized_model, 
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        output_onnx_path,
        export_params=True,
        opset_version=14,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"},
                      "attention_mask": {0: "batch_size", 1: "sequence_length"}}
    )
    print("QAT Export Complete. Ready for Android Assets Folder.")

if __name__ == "__main__":
    print("=== Android QAT Setup Script ===")
    
    local_model_name = "scam-classifier-model-transcript-lora"
    local_model_dir = f"./{local_model_name}"
    
    # Check if the model is already sitting on the Kaggle server locally
    if os.path.exists(local_model_dir) and os.path.exists(os.path.join(local_model_dir, "config.json")):
        print(f"Found local model at {local_model_dir}! Bypassing DagsHub download.")
    else:
        print(f"Local model not found. Attempting to fetch from DagsHub MLflow...")
        try:
            RUN_ID = "62ffeee7d6d446babb855d5c4af082ce" 
            local_model_dir = fetch_model_from_dagshub(RUN_ID, "./downloads")
        except Exception as e:
            print(f"\n[ERROR] MLflow Download Failed: {e}")
            print("DagsHub servers might be experiencing a 500 timeout, or your Kaggle server is missing credentials.")
            exit(1)
            
    # 2. Configure QAT
    qat_model = configure_qat(local_model_dir)
    
    # 3. ACTUAL QAT TRAINING LOOP
    print("\n--- Starting Quantization-Aware Training (1 Epoch) ---")
    
    # Locate the downloaded dataset (it was logged as a CSV in MLflow)
    import pandas as pd
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, AutoTokenizer, DataCollatorWithPadding
    
    csv_path = None
    for root, dirs, files in os.walk("./downloads"):
        for file in files:
            if file.endswith(".csv") and "train" in file.lower():
                csv_path = os.path.join(root, file)
                break
        if csv_path:
            break
            
    if csv_path is None:
        # Fallback to test/any csv if train isn't specifically named
        for root, dirs, files in os.walk("./downloads"):
            for file in files:
                if file.endswith(".csv"):
                    csv_path = os.path.join(root, file)
                    break
            if csv_path:
                break

    if csv_path is None:
        print("[ERROR] Could not find any .csv dataset in the downloaded artifacts!")
        exit(1)
        
    print(f"Loading Phase 2 Dataset from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # We only need a small sample (e.g. 500 rows) for QAT to adjust weights, no need to train on the whole thing!
    # This prevents the QAT loop from taking hours on Kaggle.
    df = df.sample(n=min(500, len(df)), random_state=42).reset_index(drop=True)
    dataset = Dataset.from_pandas(df)
    
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    
    print("Tokenizing dataset...")
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512)
    
    dataset = dataset.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./qat_checkpoints",
        num_train_epochs=1,          # 1 epoch is sufficient for QAT weight adjustment
        per_device_train_batch_size=16,
        learning_rate=2e-5,          # Very low learning rate so we don't destroy Phase 2 knowledge
        logging_steps=50,
        save_strategy="no",          # We only care about the final ONNX export
        report_to="none"
    )
    
    trainer = Trainer(
        model=qat_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("Executing Trainer.train() with FakeQuantize nodes active...")
    trainer.train()
    
    # 4. Export
    print("\n--- Training Complete! Proceeding to Export ---")
    export_qat_to_android(qat_model)


