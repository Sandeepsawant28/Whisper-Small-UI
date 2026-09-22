"""
Whisper Small Konkani Model Server
Serves the fine-tuned Konkani Whisper model (LoRA adapter on openai/whisper-small)
for real-time speech transcription via HTTP POST /transcribe.

Loads the adapter repo directly (sandeepsawant28/whisper-small-konkani) instead of
the merged repo, matching the exact loading path used in the evaluation notebook
that was already confirmed to work well.
"""
import os
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '30'

import os
import gc
import tempfile
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

BASE_MODEL_ID = "openai/whisper-small"
ADAPTER_ID = "sandeepsawant28/whisper-small-konkani"  # LoRA adapter repo (verified public model card)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Use float16 on CPU to cut memory usage roughly in half (needed for low-RAM hosting).
MODEL_DTYPE = torch.float16

model = None
processor = None

try:
    print(f"Loading Konkani Whisper Small model (LoRA adapter) on {device}...")
    processor = WhisperProcessor.from_pretrained(ADAPTER_ID)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=MODEL_DTYPE,
        low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID).to(device)
    model = model.half()  # ensure LoRA adapter weights are also float16
    model.eval()

    gc.collect()
    print("[OK] Model loaded successfully!")
except Exception as e:
    print(f"Notice: Model load failed ({e}). Running server in API ready mode.")
    traceback.print_exc()
    model = None

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin requests from web UI


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Receives audio file from UI (webm/wav/mp3), transcribes using the
    Konkani Whisper LoRA model. Whisper has no native Konkani ('kok') language
    token, so 'marathi' is used as the closest supported substitute — matching
    how the model was fine-tuned. The output text itself is genuine Konkani.
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_webm:
        audio_file.save(temp_webm.name)
        webm_path = temp_webm.name

    wav_path = webm_path.replace(".webm", ".wav")

    try:
        # Ignore empty or tiny audio fragments (< 600 bytes)
        if os.path.getsize(webm_path) < 600:
            return jsonify({
                "status": "empty",
                "text": "",
                "language": "Konkani (kok)"
            })

        # Explicitly convert webm -> wav via ffmpeg (through pydub) before transcribing.
        # Whisper expects 16kHz mono input.
        try:
            audio = AudioSegment.from_file(webm_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(wav_path, format="wav")
        except Exception as ffmpeg_err:
            print(f"[Notice] Audio chunk decoding skipped ({ffmpeg_err})")
            return jsonify({
                "status": "warning",
                "text": "",
                "warning": "Audio chunk could not be decoded by ffmpeg, skipped."
            }), 200

        if model is None or processor is None:
            return jsonify({
                "status": "warning",
                "is_mock": True,
                "error": "Whisper model not loaded on server. Check server startup logs.",
                "language": "Konkani (kok)"
            }), 503

        import librosa
        audio_input, sr = librosa.load(wav_path, sr=16000)
        if len(audio_input) < 1600:  # Less than 0.1s of audio
            return jsonify({
                "status": "empty",
                "text": "",
                "language": "Konkani (kok)"
            })

        input_features = processor(
            audio_input, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device, dtype=MODEL_DTYPE)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language="marathi",  # closest supported substitute for Konkani
                task="transcribe"
            )

        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

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


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online" if model is not None else "model_not_loaded",
        "model": "Whisper Small Konkani (LoRA adapter)",
        "backend": "PyTorch / Transformers / PEFT"
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n[RUNNING] Whisper Konkani Model API running at http://0.0.0.0:{port}/transcribe")
    app.run(host='0.0.0.0', port=port, debug=False)