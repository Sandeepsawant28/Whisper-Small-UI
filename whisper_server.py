"""
Whisper Small Konkani (CTranslate2 int8) Model Server
Serves the quantized model via faster-whisper for fast CPU inference.
HTTP POST /transcribe — same API contract as the earlier Flask server.
"""
import os
import gc
import tempfile
import traceback
import threading
import subprocess

from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

HF_REPO_ID = "sandeepsawant28/whisper-small-konkani-numbers"

model = None
transcribe_lock = threading.Lock()  # prevents overlapping requests from stacking memory

try:
    print(f"Downloading/locating model files for {HF_REPO_ID}...")
    local_path = snapshot_download(HF_REPO_ID)
    ct2_path = os.path.join(local_path, "ct2")

    print(f"Loading CTranslate2 int8 model from {ct2_path}...")
    # cpu_threads=1 keeps memory usage low and predictable on RAM-constrained hosts
    model = WhisperModel(ct2_path, device="cpu", compute_type="int8", cpu_threads=2)
    print("[OK] Model loaded successfully on CPU (int8).")
except Exception as e:
    print(f"Notice: Model load failed ({e}).")
    traceback.print_exc()
    model = None

app = Flask(__name__)
CORS(app)


def convert_to_wav(webm_path, wav_path):
    """Direct ffmpeg subprocess call — avoids pydub's in-Python memory copy."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", webm_path,
            "-ar", "16000", "-ac", "1",
            "-f", "wav", wav_path
        ],
        capture_output=True,
        timeout=15
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="ignore"))


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    with transcribe_lock:  # serialize requests so memory never stacks
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_webm:
            audio_file.save(temp_webm.name)
            webm_path = temp_webm.name

        wav_path = webm_path.replace(".webm", ".wav")
        segments, info = None, None

        try:
            if os.path.getsize(webm_path) < 600:
                return jsonify({"status": "empty", "text": "", "language": "Konkani (kok)"})

            try:
                convert_to_wav(webm_path, wav_path)
            except Exception as ffmpeg_err:
                return jsonify({
                    "status": "warning",
                    "text": "",
                    "warning": f"Audio decoding failed: {ffmpeg_err}"
                }), 200

            if model is None:
                return jsonify({
                    "status": "warning",
                    "is_mock": True,
                    "error": "Model not loaded on server.",
                    "language": "Konkani (kok)"
                }), 503

            segments, info = model.transcribe(
                wav_path,
                language="mr",   # Marathi as closest MMS/Whisper substitute for Konkani
                beam_size=1,      # default is 5 — beam search multiplies memory/compute
                best_of=1,
                vad_filter=False,
            )
            text = " ".join([seg.text for seg in segments]).strip()

            if not text:
                return jsonify({"status": "empty", "text": "", "language": "Konkani (kok)"})

            return jsonify({
                "status": "success",
                "is_mock": False,
                "text": text,
                "language": "Konkani (kok)"
            })

        except Exception as err:
            traceback.print_exc()
            return jsonify({"error": str(err)}), 500
        finally:
            for p in (webm_path, wav_path):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            try:
                del segments, info
            except NameError:
                pass
            gc.collect()  # free memory back to the OS after each request


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online" if model is not None else "model_not_loaded",
        "model": "Whisper Small Konkani Numbers (CTranslate2 int8)",
        "backend": "faster-whisper / CTranslate2",
        "hardware": "CPU"
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n[RUNNING] Whisper Konkani (CT2) API running at http://0.0.0.0:{port}/transcribe")
    app.run(host='0.0.0.0', port=port, debug=False)
