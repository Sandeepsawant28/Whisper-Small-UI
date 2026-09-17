import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import librosa

BASE_MODEL_ID = "openai/whisper-small"
ADAPTER_ID = "sandeepsawant28/whisper-small-konkani"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")
processor = WhisperProcessor.from_pretrained(ADAPTER_ID)
base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID)
model = PeftModel.from_pretrained(base_model, ADAPTER_ID).to(device)
model.eval()

audio_input, sr = librosa.load("test_audio.wav", sr=16000)
print(f"Audio loaded, length: {len(audio_input)}, sample rate: {sr}")

input_features = processor(
    audio_input, sampling_rate=16000, return_tensors="pt"
).input_features.to(device)

print("Starting generation...")
with torch.no_grad():
    predicted_ids = model.generate(
        input_features,
        language="marathi",
        task="transcribe"
    )

text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(f"Transcription result: '{text}'")
