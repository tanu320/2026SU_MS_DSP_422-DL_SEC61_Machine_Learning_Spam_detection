import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.quantization
import mlflow
import os

def fetch_model_from_dagshub(run_id: str, artifact_path: str, download_dir: str):
    """
    Downloads the Phase 2 (Transcript Retrained) model from DagsHub MLflow.
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
    
    # Robust file-by-file download to bypass DagsHub 500 timeout on large directory zipping
    def download_recursive(path, dest_dir):
        artifacts = client.list_artifacts(run_id, path)
        for artifact in artifacts:
            if artifact.is_dir:
                # If there are checkpoints, we only really need the final model files.
                # However, to be safe, we will recursively download all dirs
                download_recursive(artifact.path, dest_dir)
            else:
                print(f"  Downloading {artifact.path}...")
                client.download_artifacts(run_id, artifact.path, dest_dir)
                
    print("Starting robust file-by-file download...")
    download_recursive(artifact_path, download_dir)
    
    local_path = os.path.join(download_dir, artifact_path)
    print(f"Model successfully downloaded to: {local_path}")
    return local_path

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
    if os.path.exists(local_model_dir):
        print(f"Found local model at {local_model_dir}! Bypassing DagsHub download.")
    else:
        print(f"Local model not found. Attempting to fetch from DagsHub MLflow...")
        try:
            RUN_ID = "62ffeee7d6d446babb855d5c4af082ce" 
            local_model_dir = fetch_model_from_dagshub(RUN_ID, local_model_name, "./downloads")
        except Exception as e:
            print(f"\n[ERROR] MLflow Download Failed: {e}")
            print("DagsHub servers might be experiencing a 500 timeout, or your Kaggle server is missing the MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD environment variables.")
            print(f"Please either set those credentials, or manually ensure the '{local_model_name}' folder is uploaded to your Kaggle working directory.\n")
            exit(1)
            
    # 2. Configure QAT
    qat_model = configure_qat(local_model_dir)
    
    # 3. ---> RUN YOUR PYTORCH TRAINING LOOP HERE FOR 1 EPOCH ON PHASE 2 DATA <---
    print("\n[!] Please insert your Phase 2 dataset loading and Trainer.train() loop here before exporting!\n")
    
    # 4. Export
    # export_qat_to_android(qat_model)


