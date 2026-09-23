FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model at BUILD time, not runtime
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('sandeepsawant28/whisper-small-konkani-numbers')"

COPY whisper_server.py .

EXPOSE 5000

CMD ["python", "whisper_server.py"]
