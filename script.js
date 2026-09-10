const WHISPER_URL = "http://localhost:5000/transcribe";
const CHUNK_MS = 4000; // send audio to server every 4 seconds

let mediaRecorder = null;
let stream = null;
let isRecording = false;

const btnRecord = document.getElementById("btn-record");
const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");

btnRecord.addEventListener("click", () => {
  if (!isRecording) {
    startRecording();
  } else {
    stopRecording();
  }
});

async function startRecording() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    statusEl.textContent = "Microphone access denied or unavailable.";
    console.error(err);
    return;
  }

  mediaRecorder = new MediaRecorder(stream);

  mediaRecorder.ondataavailable = async (e) => {
    if (e.data.size === 0) return;
    await sendChunkToServer(e.data);
  };

  mediaRecorder.start(CHUNK_MS); // fires ondataavailable every CHUNK_MS
  isRecording = true;
  btnRecord.textContent = "⏹ Stop Recording";
  btnRecord.classList.add("recording");
  statusEl.textContent = "Recording... speak now";
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }
  isRecording = false;
  btnRecord.textContent = "🎤 Start Recording";
  btnRecord.classList.remove("recording");
  statusEl.textContent = "Stopped";
}

async function sendChunkToServer(blob) {
  const formData = new FormData();
  formData.append("audio", blob, "chunk.webm");

  statusEl.textContent = "Transcribing...";

  try {
    const res = await fetch(WHISPER_URL, {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.error || `HTTP ${res.status}`);
    }

    const data = await res.json();
    if (data.text && data.text.trim().length > 0) {
      appendTranscript(data.text.trim());
    }
    statusEl.textContent = isRecording ? "Recording... speak now" : "Stopped";
  } catch (err) {
    console.error("Transcription failed:", err);
    statusEl.textContent = `Error: ${err.message}`;
  }
}

function appendTranscript(text) {
  const p = document.createElement("p");
  p.textContent = text;
  transcriptEl.appendChild(p);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}