import os
import json
import time
import numpy as np
import gradio as gr
from dotenv import load_dotenv

LIVE_WINDOW_SEC = 3.0

# We rely on configs/inference_config.json for the backend selection.
config_path = "configs/inference_config.json"
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

# Load pipeline (will auto-download models from DagsHub if DagsHub secrets are set in HF Spaces)
print("Initializing configured inference pipeline...")
import importlib
infer_module = importlib.import_module("src.evaluation.inference_pipeline")
InferencePipeline = infer_module.InferencePipeline
pipeline = None
pipeline_error = None
try:
    pipeline = InferencePipeline()
    backend_label = f"ASR={pipeline.asr_backend.upper()} | Classifier={pipeline.clf_backend.upper()}"
except Exception as exc:
    pipeline_error = exc
    backend_label = f"Initialization failed: {exc}"
    print(f"[STARTUP ERROR] Failed to initialize inference pipeline: {exc}", flush=True)

def _display_prediction(prediction):
    """The pipeline returns raw 0/1 (or an already-worded string for the
    empty-audio case) -- normalize to the label string the UI renders."""
    if prediction == 1:
        return "Scam"
    if prediction == 0:
        return "Legitimate"
    return prediction

def process_audio(audio_file_path):
    if not audio_file_path:
        return "No audio provided.", "N/A", "N/A"

    if pipeline is None:
        return f"Pipeline initialization failed: {pipeline_error}", "Error", "Error"

    try:
        # Run inference using the CPU GGUF pipeline + Logistic Regression head
        result = pipeline.process_audio(audio_file_path)

        transcript = result.get("transcript", "")
        prediction = _display_prediction(result.get("prediction", "Unknown"))
        metrics = result.get("metrics", {})

        latency_str = (f"ASR: {metrics.get('asr_latency', 0):.2f}s | "
                       f"Classifier: {metrics.get('classifier_latency', 0):.2f}s | "
                       f"Total: {metrics.get('total_latency', 0):.2f}s")

        return transcript, prediction, latency_str

    except Exception as e:
        return f"Error: {str(e)}", "Error", "Error"

def _alert_html(prediction):
    """Render a colored status banner for the live monitor."""
    if not prediction:
        return (
            "<div style='padding:16px;border-radius:8px;background:#374151;"
            "color:white;text-align:center;font-size:1.15em;'>"
            "🎙️ Waiting for speech...</div>"
        )
    if str(prediction).lower().startswith("scam"):
        return (
            "<div style='padding:16px;border-radius:8px;background:#dc2626;"
            "color:white;text-align:center;font-size:1.3em;font-weight:bold;'>"
            f"🚨 SCAM ALERT — {prediction}</div>"
        )
    return (
        "<div style='padding:16px;border-radius:8px;background:#16a34a;"
        "color:white;text-align:center;font-size:1.15em;'>"
        f"✅ {prediction}</div>"
    )

def live_monitor(new_chunk, state):
    """Called on every audio chunk pushed by the browser's mic stream.

    Buffers raw samples until a full LIVE_WINDOW_SEC window is available,
    transcribes just that window, appends the text to the running
    transcript, then re-classifies the *entire* transcript so far.
    """
    if new_chunk is None:
        state = state or {}
        return (
            state,
            state.get("transcript", ""),
            _alert_html(state.get("last_prediction")),
            "Idle — start recording to begin monitoring.",
        )

    if pipeline is None:
        return state, "", _alert_html(None), f"Pipeline initialization failed: {pipeline_error}"

    sr, y = new_chunk
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.reshape(-1)

    if state is None or state.get("sr") != sr:
        state = {"buffer": np.zeros(0, dtype=y.dtype), "sr": sr, "transcript": "",
                  "last_prediction": None, "window_count": 0}

    state["buffer"] = np.concatenate([state["buffer"], y])
    window_samples = int(LIVE_WINDOW_SEC * sr)

    if len(state["buffer"]) < window_samples:
        remaining = (window_samples - len(state["buffer"])) / sr
        status = f"Buffering... {remaining:.1f}s until next check"
        return state, state["transcript"] or "(listening...)", _alert_html(state["last_prediction"]), status

    # Pull out exactly one window's worth of audio; keep any leftover for next round.
    window_audio = state["buffer"][:window_samples]
    state["buffer"] = state["buffer"][window_samples:]

    try:
        t0 = time.time()
        chunk_text = pipeline.transcribe_chunk(sr, window_audio).strip()
        asr_latency = time.time() - t0

        if chunk_text:
            state["transcript"] = (state["transcript"] + " " + chunk_text).strip()

        t1 = time.time()
        if state["transcript"]:
            prediction = _display_prediction(pipeline.classify(state["transcript"]))
        else:
            prediction = "Legitimate (No speech yet)"
        clf_latency = time.time() - t1

        state["last_prediction"] = prediction
        state["window_count"] += 1
        status = (f"Window #{state['window_count']} | ASR: {asr_latency:.2f}s | "
                   f"Classifier: {clf_latency:.2f}s")
    except Exception as e:
        status = f"Error analyzing window: {e}"

    return state, state["transcript"] or "(listening...)", _alert_html(state["last_prediction"]), status

def reset_live_state():
    return None, "", _alert_html(None), "Idle — start recording to begin monitoring."

def reset_upload_state():
    return None, "", "", ""

# Build Gradio UI
with gr.Blocks(title="Scam Detection AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Scam Detection AI")
    gr.Markdown(f"Active pipeline: `{backend_label}`.")

    with gr.Tabs():
        with gr.Tab("📁 Upload a Recording"):
            gr.Markdown("Upload an audio recording of a phone call.")

            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(type="filepath", label="Upload Audio Call (.wav, .mp3)")
                    with gr.Row():
                        analyze_btn = gr.Button("Analyze Intent", variant="primary")
                        clear_upload_btn = gr.Button("Clear")

                with gr.Column():
                    prediction_output = gr.Textbox(label="AI Prediction (Scam vs Legitimate)", lines=1)
                    transcript_output = gr.Textbox(label="Whisper Transcript", lines=5)
                    latency_output = gr.Textbox(label="Inference Latency", lines=1)

            analyze_btn.click(
                fn=process_audio,
                inputs=audio_input,
                outputs=[transcript_output, prediction_output, latency_output]
            )
            clear_upload_btn.click(
                fn=reset_upload_state,
                inputs=None,
                outputs=[audio_input, transcript_output, prediction_output, latency_output],
            )

        with gr.Tab("🔴 Live Call Monitor"):
            gr.Markdown(
                f"Start recording and speak. Every **{LIVE_WINDOW_SEC:.0f} seconds** of new "
                "audio is transcribed and appended to the running transcript, which is then "
                "re-checked for scam intent — no need to wait for the call to end."
            )

            live_state = gr.State(None)

            with gr.Row():
                with gr.Column():
                    mic_input = gr.Audio(
                        sources=["microphone"],
                        streaming=True,
                        label="Live Microphone Feed",
                    )
                    clear_btn = gr.Button("Clear / Reset Call")

                with gr.Column():
                    alert_output = gr.HTML(_alert_html(None))
                    live_transcript_output = gr.Textbox(label="Running Transcript", lines=6)
                    live_status_output = gr.Textbox(label="Status", lines=1)

            mic_input.stream(
                fn=live_monitor,
                inputs=[mic_input, live_state],
                outputs=[live_state, live_transcript_output, alert_output, live_status_output],
            )

            clear_btn.click(
                fn=reset_live_state,
                inputs=None,
                outputs=[live_state, live_transcript_output, alert_output, live_status_output],
            )

# Streaming mic input pushes one update per chunk; queue() runs those through
# a worker so a slow ASR/classify pass on one window doesn't block the rest
# of the UI (including the upload tab) while it's in flight.
demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
