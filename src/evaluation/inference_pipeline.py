import os
import json
import re
import subprocess
import tempfile
import time
import numpy as np

class InferencePipeline:
    # Modern (CMake) whisper.cpp builds put the CLI under build/bin/; older
    # `make`-based builds put it at the repo root. Check every location so
    # an image that already compiled it during a Docker build step isn't
    # rebuilt on every cold start.
    WHISPER_BIN_PATHS = [
        "./whisper.cpp/build/bin/whisper-cli",
        "./whisper.cpp/build/bin/main",
        "./whisper.cpp/bin/whisper-cli",
        "./whisper.cpp/bin/main",
        "./whisper.cpp/whisper-cli",
        "./whisper.cpp/main",
    ]

    def __init__(self, config_path_or_dict="configs/inference_config.json"):
        if isinstance(config_path_or_dict, dict):
            self.config = config_path_or_dict
        else:
            with open(config_path_or_dict, 'r') as f:
                self.config = json.load(f)
        
        self.clf_backend = self.config.get("classifier_backend", self.config.get("backend", "fp16"))
        self.asr_backend = self.config.get("asr_backend", self.config.get("backend", "fp16"))
        
        print(f"Initializing Pipeline | CLF: {self.clf_backend.upper()} | ASR: {self.asr_backend.upper()}")
        
        if self.clf_backend == "fp16":
            self._init_fp16_classifier()
        elif self.clf_backend == "gguf":
            self._init_gguf_classifier()
        else:
            raise ValueError(f"Unknown classifier backend: {self.clf_backend}")
            
        if self.asr_backend == "fp16":
            self._init_fp16_asr()
        elif self.asr_backend == "gguf":
            self._init_gguf_asr()
        else:
            raise ValueError(f"Unknown ASR backend: {self.asr_backend}")

    def _init_fp16_asr(self):
        import torch
        from transformers import pipeline
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load ASR
        print(f"Loading FP16 ASR: {self.config.get('fp16_asr_model_name', 'openai/whisper-tiny')}")
        self.asr_pipe = pipeline("automatic-speech-recognition", 
                               model=self.config.get("fp16_asr_model_name", "openai/whisper-tiny"), 
                               device=self.device,
                               chunk_length_s=30)
                               
    def _init_fp16_classifier(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Classifier
        model_name = self.config['fp16_classifier_model_name']
        print(f"Loading FP16 Classifier: {model_name}")
        
        if model_name.startswith("models:/"):
            import mlflow
            print("Downloading and loading model from MLflow registry...")
            components = mlflow.transformers.load_model(model_name, return_type="components")
            self.tokenizer = components["tokenizer"]
            self.classifier = components["model"].to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.classifier = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

    def _init_gguf_classifier(self):
        from llama_cpp import Llama
        
        # Load Classifier (GGUF)
        model_path = self.config['classifier_model_path']
        if not os.path.exists(model_path):
            print(f"Downloading {model_path} from DagsHub...")
            self._download_model_artifact(
                model_path,
                [
                    "artifacts/refactored_pipeline/06_ptq_modernbert/gguf",
                    "artifacts/06_ptq_modernbert/gguf",
                    "models/gguf_classifier",
                    "artifacts/feature/phase-2-audio-asr/gguf",
                ],
            )
            
        print(f"Loading GGUF Classifier: {model_path}")
        self.llm = Llama(model_path=model_path, verbose=False, embedding=True, n_ctx=1024)
        
        # Load the custom trained Scikit-Learn classification head
        import joblib
        head_path = self.config.get("gguf_classifier_head_path", "models/gguf/gguf_classifier_head.joblib")
        if not os.path.exists(head_path):
            print(f"Downloading {head_path} from DagsHub...")
            self._download_from_dagshub_any(
                [
                    head_path,
                    "artifacts/refactored_pipeline/06_ptq_modernbert/gguf/gguf_classifier_head.joblib",
                    "artifacts/06_ptq_modernbert/gguf/gguf_classifier_head.joblib",
                    "models/gguf/gguf_classifier_head.joblib",
                    "artifacts/feature/phase-2-audio-asr/gguf/gguf_classifier_head.joblib",
                    "artifacts/feature/phase-3.5-benchmark/gguf_classifier_head.joblib",
                    "artifacts/feature/phase-3.5-benchmark/gguf/gguf_classifier_head.joblib",
                ],
                head_path,
            )
        if not os.path.exists(head_path):
            raise RuntimeError(
                "CRITICAL ERROR: GGUF Classification Head not found. "
                f"Expected path: {head_path}"
            )
        print(f"Loading GGUF Classification Head: {head_path}")
        self.gguf_head = joblib.load(head_path)

    def _init_gguf_asr(self):
        self.whisper_model_path = self.config['asr_model_path']
        if not os.path.exists(self.whisper_model_path):
            print(f"Downloading {self.whisper_model_path} from DagsHub...")
            self._download_model_artifact(
                self.whisper_model_path,
                [
                    "artifacts/refactored_pipeline/06_ptq_modernbert/ggml",
                    "artifacts/06_ptq_modernbert/ggml",
                    "models/ggml_whisper",
                    "artifacts/feature/phase-2-audio-asr/ggml",
                ],
            )

        # ASR runs via the whisper.cpp CLI over subprocess rather than the
        # pywhispercpp Python bindings: pywhispercpp's PyPI sdist doesn't
        # vendor whisper.cpp's own sources (no ggml.h), so it cannot be
        # built from source on Linux at all -- only a prebuilt macOS wheel
        # exists. Building whisper.cpp itself (this is the real, actively
        # maintained project) works identically on every platform.
        if self._find_whisper_binary() is None:
            print("whisper.cpp not compiled. Building now...", flush=True)
            if not os.path.exists("./whisper.cpp"):
                subprocess.run(
                    ["git", "clone", "https://github.com/ggml-org/whisper.cpp.git"],
                    check=True,
                )
            subprocess.run(
                ["cmake", "-B", "build"], cwd="./whisper.cpp", check=True
            )
            subprocess.run(
                ["cmake", "--build", "build", "--config", "Release", "-j"],
                cwd="./whisper.cpp",
                check=True,
            )
        print(f"Using whisper.cpp CLI: {self._find_whisper_binary()}", flush=True)

    def _find_whisper_binary(self):
        for p in self.WHISPER_BIN_PATHS:
            if os.path.exists(p):
                return p
        return None

    def _download_from_dagshub(self, file_path):
        import boto3
        from dotenv import load_dotenv
        load_dotenv()
        owner = os.getenv("DAGSHUB_REPO_OWNER", "kureeltanishq")
        name = os.getenv("DAGSHUB_REPO_NAME", "2026SU_MS_DSP_422-DL_SEC61_Machine_Learning_Spam_detection")
        token = os.getenv("MLFLOW_TRACKING_PASSWORD")
        
        import dagshub
        if token:
            dagshub.auth.add_app_token(token)
            
        s3 = dagshub.get_repo_bucket_client(f"{owner}/{name}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        s3.download_file(name, file_path, file_path)

    def _download_model_artifact(self, local_path, remote_prefixes):
        filename = os.path.basename(local_path)
        remote_paths = [f"{prefix}/{filename}" for prefix in remote_prefixes] + [local_path]
        self._download_from_dagshub_any(remote_paths, local_path)

    def _is_valid_downloaded_artifact(self, local_path):
        if not os.path.exists(local_path):
            return False

        size_bytes = os.path.getsize(local_path)
        if size_bytes < 1024:
            return False

        if local_path.endswith(".gguf"):
            with open(local_path, "rb") as f:
                return f.read(4) == b"GGUF"

        if local_path.endswith(".bin"):
            # whisper.cpp GGML binaries begin with a GGML/GGMF/GGJT magic or
            # equivalent little-endian integer. Size catches pointer/html files.
            return size_bytes > 1_000_000

        if local_path.endswith(".joblib"):
            return size_bytes > 1024

        return True

    def _download_from_dagshub_any(self, remote_paths, local_path):
        import dagshub
        from dotenv import load_dotenv
        load_dotenv()
        owner = os.getenv("DAGSHUB_REPO_OWNER", "kureeltanishq")
        name = os.getenv("DAGSHUB_REPO_NAME", "2026SU_MS_DSP_422-DL_SEC61_Machine_Learning_Spam_detection")
        token = os.getenv("MLFLOW_TRACKING_PASSWORD")

        if token:
            dagshub.auth.add_app_token(token)

        s3 = dagshub.get_repo_bucket_client(f"{owner}/{name}")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        last_error = None
        for remote_path in remote_paths:
            try:
                s3.download_file(name, remote_path, local_path)
                if self._is_valid_downloaded_artifact(local_path):
                    print(
                        f"Downloaded {remote_path} -> {local_path} "
                        f"({os.path.getsize(local_path) / (1024 * 1024):.2f} MB)",
                        flush=True,
                    )
                    return
                last_error = RuntimeError(f"Downloaded invalid artifact from {remote_path}")
                try:
                    os.remove(local_path)
                except OSError:
                    pass
            except Exception as exc:
                last_error = exc
        raise FileNotFoundError(
            f"Could not download {local_path} from any known DagsHub path. "
            f"Tried {remote_paths}. Last error: {last_error}"
        )

    def process_audio(self, audio_file):
        metrics = {}
        
        # --- 1. ASR Phase ---
        t0 = time.time()
        transcript = self._transcribe(audio_file)
        metrics['asr_latency'] = time.time() - t0
        
        # --- 2. Classifier Phase ---
        t1 = time.time()
        prediction = self._classify(transcript)
        metrics['classifier_latency'] = time.time() - t1
        
        metrics['total_latency'] = metrics['asr_latency'] + metrics['classifier_latency']
        
        return {
            "transcript": transcript,
            "prediction": prediction,
            "metrics": metrics
        }

    def classify(self, text):
        """Public wrapper so callers (e.g. the live-monitor UI/API) can
        classify arbitrary/cumulative text without reaching into _classify."""
        return self._classify(text)

    def transcribe_chunk(self, sr, audio_array):
        """Transcribe a raw in-memory audio chunk (e.g. a live-mic window).

        audio_array: 1-D numpy array (int16 or float) at sample rate `sr`.
        Writes it to a temp wav and reuses the normal file-based ASR path.
        """
        import soundfile as sf

        audio_array = np.asarray(audio_array)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        if audio_array.dtype.kind == "f":
            # float samples are expected in [-1, 1]
            audio_array = np.clip(audio_array, -1.0, 1.0)
        elif audio_array.dtype != np.int16:
            audio_array = audio_array.astype(np.int16)

        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        try:
            sf.write(temp_wav, audio_array, sr, subtype="PCM_16")
            # A silent window is expected during normal call pauses, unlike a
            # fully silent uploaded recording -- don't raise for it here.
            return self._transcribe(temp_wav, allow_empty=True)
        finally:
            try:
                os.remove(temp_wav)
            except OSError:
                pass

    def _clean_asr_transcript(self, transcript):
        """Normalize whisper.cpp placeholder outputs from silent mic chunks."""
        if not transcript:
            return ""

        text = str(transcript).strip()
        text = re.sub(r"(?i)\[?\(?\s*blank[_\s-]*audio\s*\)?\]?", " ", text)
        text = re.sub(r"(?i)<\|no[_\s-]*speech\|>", " ", text)

        # If a line only had timestamps and a blank/no-speech marker, the
        # marker removal can leave timestamp punctuation behind. Drop those
        # remnants instead of showing them in the live transcript.
        cleaned_lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if re.fullmatch(r"[\[\]\(\)\d:.,\-\s>]+", line):
                continue
            cleaned_lines.append(line)

        return " ".join(" ".join(cleaned_lines).split()).strip()

    def _transcribe(self, audio_file, allow_empty=False):
        if self.asr_backend == "fp16":
            result = self.asr_pipe(audio_file)
            transcript = self._clean_asr_transcript(result["text"])
            if not transcript and not allow_empty:
                error_msg = "Whisper ASR returned an empty transcript."
                print(f"[ASR ERROR] {error_msg}", flush=True)
                raise RuntimeError(error_msg)
            return transcript
        else:
            # whisper.cpp's CLI only accepts 16kHz mono PCM WAV -- re-encode
            # whatever format we were handed (mp3, webm/opus mic chunks, etc.)
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            try:
                ffmpeg_result = subprocess.run(
                    ["ffmpeg", "-y", "-i", audio_file,
                     "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", temp_wav],
                    capture_output=True, text=True,
                )
                if ffmpeg_result.returncode != 0:
                    raise RuntimeError(f"ffmpeg audio conversion failed: {ffmpeg_result.stderr[-1000:]}")

                whisper_bin = self._find_whisper_binary()
                if not whisper_bin:
                    raise FileNotFoundError("Could not locate compiled whisper-cli/main binary in whisper.cpp/")

                out_base = tempfile.NamedTemporaryFile(prefix="whisper_transcript_", delete=True).name
                out_txt = f"{out_base}.txt"
                cmd = [
                    whisper_bin,
                    "-m", self.whisper_model_path,
                    "-f", temp_wav,
                    "-nt",
                    "-sns",
                    "-otxt",
                    "-of", out_base,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)

                transcript = ""
                if os.path.exists(out_txt):
                    with open(out_txt, "r", encoding="utf-8") as f:
                        transcript = f.read().strip()
                    os.remove(out_txt)
                if not transcript:
                    transcript = result.stdout.strip()
                transcript = self._clean_asr_transcript(transcript)

                if result.returncode != 0:
                    raise RuntimeError(f"whisper.cpp failed: {result.stderr[-1500:]}")

                if not transcript and not allow_empty:
                    error_msg = "Whisper ASR returned an empty transcript."
                    print(f"[ASR ERROR] {error_msg}", flush=True)
                    raise RuntimeError(error_msg)

                return transcript
            finally:
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass

    def _classify(self, text):
        if self.clf_backend == "fp16":
            import torch
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.classifier(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_idx = torch.argmax(probs, dim=1).item()
                # Assuming 1 is scam, 0 is legit
                return 1 if pred_idx == 1 else 0
        else:
            # GGUF ModernBERT classification using embeddings
            text = text.strip()
            if not text:
                return "Legitimate (Empty Audio)"
                
            # We extract the embeddings and mean-pool them across the sequence dimension
            # We truncate to 5000 characters to support up to ~5 minutes of audio 
            # while maintaining fast CPU inference speeds.
            import time
            t_emb_start = time.time()
            raw_emb = self.llm.embed(text[:5000])
            emb_latency = time.time() - t_emb_start
            
            arr = np.array(raw_emb)
            print(f"[CLASSIFIER] Truncated text to {len(text[:5000])} chars. Embedding shape: {arr.shape} | Extraction latency: {emb_latency:.2f}s", flush=True)
            
            # Robust pooling depending on llama-cpp-python return shape
            if arr.ndim == 3:
                embeds = np.mean(arr[0], axis=0)  # (1, seq, hidden) -> (seq, hidden) -> (hidden,)
            elif arr.ndim == 2:
                embeds = np.mean(arr, axis=0)     # (seq, hidden) or (1, hidden) -> (hidden,)
            else:
                embeds = arr                      # (hidden,)
                
            
            # Scikit-learn expects 2D array: (n_samples, n_features)
            pred_idx = self.gguf_head.predict([embeds])[0]
            return 1 if pred_idx == 1 else 0
            
    def process_batch(self, audio_files, batch_size=8):
        metrics = {}
        t0 = time.time()
        
        # --- 1. ASR Phase ---
        transcripts = self._transcribe_batch(audio_files, batch_size=batch_size)
        metrics['asr_latency'] = time.time() - t0
        
        # --- 2. Classifier Phase ---
        t1 = time.time()
        predictions = self._classify_batch(transcripts, batch_size=batch_size)
        metrics['classifier_latency'] = time.time() - t1
        
        metrics['total_latency'] = metrics['asr_latency'] + metrics['classifier_latency']
        
        results = []
        for t, p in zip(transcripts, predictions):
            results.append({
                "transcript": t,
                "prediction": p,
                "metrics": {
                    "asr_latency": metrics['asr_latency'] / len(audio_files),
                    "classifier_latency": metrics['classifier_latency'] / len(audio_files),
                    "total_latency": metrics['total_latency'] / len(audio_files)
                }
            })
        return results

    def _transcribe_batch(self, audio_files, batch_size=8):
        if self.asr_backend == "fp16":
            results = self.asr_pipe(audio_files, batch_size=batch_size)
            return [res["text"].strip() for res in results]
        else:
            # GGUF/whisper.cpp doesn't support native batching across multiple files easily, 
            # so we process sequentially which natively multi-threads per file anyway
            return [self._transcribe(f) for f in audio_files]

    def _classify_batch(self, texts, batch_size=8):
        if self.clf_backend == "fp16":
            import torch
            predictions = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    outputs = self.classifier(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    pred_idxs = torch.argmax(probs, dim=1).tolist()
                    predictions.extend([1 if idx == 1 else 0 for idx in pred_idxs])
            return predictions
        else:
            return [self._classify(t) for t in texts]
            
if __name__ == "__main__":
    pipeline = InferencePipeline()
    print("Pipeline ready.")
