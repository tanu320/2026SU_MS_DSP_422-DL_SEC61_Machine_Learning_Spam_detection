import mlflow
import os

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/kureeltanishq/2026SU_MS_DSP_422-DL_SEC61_Machine_Learning_Spam_detection.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "kureeltanishq"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "dd69aa5a833e4ee052c5256e4a468d35db0f639f"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

runs = mlflow.search_runs()
print("Total runs found:", len(runs))

for idx, run in runs.iterrows():
    run_id = run["run_id"]
    name = run.get("tags.mlflow.runName", "Unnamed")
    print(f"Run {run_id} ({name}):")
    
    try:
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        for artifact in artifacts:
            print(f"  - {artifact.path}")
    except Exception as e:
        print(f"  Failed to list artifacts: {e}")
