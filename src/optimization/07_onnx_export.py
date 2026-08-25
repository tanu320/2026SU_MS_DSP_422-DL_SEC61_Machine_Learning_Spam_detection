import os
import sys
import torch
import joblib
import dagshub
import mlflow
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

def install_skl2onnx():
    try:
        import skl2onnx
    except ImportError:
        import subprocess
        print("Installing skl2onnx for MLP conversion...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "skl2onnx", "onnxruntime"])

def export_modernbert_to_onnx():
    print("--- Exporting ModernBERT to ONNX ---")
    model_path = "./scam-classifier-model-transcript-lora"
    if not os.path.exists(model_path):
        print(f"Error: Base model {model_path} not found. Please ensure script 05 has run successfully.")
        return False
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dummy_text = "Hello, this is a test transcript for tracing the ONNX graph."
    inputs = tokenizer(dummy_text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    os.makedirs("models/onnx", exist_ok=True)
    onnx_path = "models/onnx/modernbert_base.onnx"
    
    print(f"Exporting PyTorch model to {onnx_path}...")
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
        }
    )
    
    print("Performing INT8 Dynamic Quantization on ONNX model...")
    import onnxruntime
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    quantized_path = "models/onnx/modernbert_base_int8.onnx"
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QUInt8
    )
    
    size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
    print(f"ONNX Model exported and quantized successfully! Size: {size_mb:.2f} MB")
    return True

def export_mlp_to_onnx():
    print("--- Exporting MLP Head to ONNX ---")
    install_skl2onnx()
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    mlp_path = "models/gguf/gguf_classifier_head.joblib"
    if not os.path.exists(mlp_path):
        print(f"Error: MLP model {mlp_path} not found. Please ensure script 06 has run successfully.")
        return False
        
    mlp_model = joblib.load(mlp_path)
    
    initial_type = [('float_input', FloatTensorType([None, 768]))]
    onnx_mlp = convert_sklearn(mlp_model, initial_types=initial_type)
    
    os.makedirs("models/onnx", exist_ok=True)
    onnx_mlp_path = "models/onnx/mlp_classifier_head.onnx"
    
    with open(onnx_mlp_path, "wb") as f:
        f.write(onnx_mlp.SerializeToString())
        
    print(f"MLP Head successfully exported to {onnx_mlp_path}")
    return True
    
def upload_to_dagshub():
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    if not repo_owner or not repo_name:
        print("DagsHub credentials not found in .env, skipping upload.")
        return
        
    print("Connecting to DagsHub MLflow to log ONNX artifacts...")
    dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
    mlflow.set_experiment("scam-detection/android_deployment")
    
    with mlflow.start_run(run_name="onnx_export_int8"):
        mlflow.log_artifact("models/onnx/modernbert_base_int8.onnx", artifact_path="android_models")
        mlflow.log_artifact("models/onnx/mlp_classifier_head.onnx", artifact_path="android_models")
        print("Successfully logged ONNX models to DagsHub MLflow.")

if __name__ == "__main__":
    if export_modernbert_to_onnx() and export_mlp_to_onnx():
        upload_to_dagshub()
