const express = require("express");
const multer = require("multer");
const cors = require("cors");
const path = require("path");

// Use Node's built-in global fetch if available (Node 18+).
// Only fall back to the node-fetch package on older Node versions.
// This avoids the "fetch is not a function" crash caused by node-fetch v3+
// being ESM-only and incompatible with require().
let fetch = global.fetch;
if (typeof fetch !== "function") {
  fetch = (...args) => import("node-fetch").then(({ default: f }) => f(...args));
}

const app = express();
app.use(cors());
app.use(express.static(__dirname)); // Serve index.html, app.js, styles.css

const upload = multer({ storage: multer.memoryStorage() });

const HF_TOKEN = process.env.HF_TOKEN;
const MODEL_REPO = "sandeepsawant28/whisper-small-konkani-merged";
// Hugging Face fully retired api-inference.huggingface.co (it now returns
// DNS ENOTFOUND / HTTP 410). All Serverless Inference calls now go through
// the new router: https://router.huggingface.co/hf-inference/models/<repo>
const API_URL = `https://router.huggingface.co/hf-inference/models/${MODEL_REPO}`;

app.get("/health", (req, res) => {
  res.json({ status: "ok", model: "Konkani Whisper Small (merged)" });
});

app.post("/transcribe", upload.single("audio"), async (req, res) => {
  try {
    if (!req.file || !req.file.buffer) {
      return res.status(400).json({ error: "No audio file provided" });
    }

    if (!HF_TOKEN) {
      console.error("HF_TOKEN is not set! Set it before starting the server, e.g.:");
      console.error('  $env:HF_TOKEN="your_token_here"   (PowerShell)');
      return res.status(500).json({ error: "Server misconfigured: HF_TOKEN is missing" });
    }

    const audioBuffer = req.file.buffer;

    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        "Content-Type": "audio/webm",
      },
      body: audioBuffer,
    });

    // Log the raw HTTP status so we can tell auth errors, model-not-found,
    // and cold-start responses apart from each other.
    console.log("HF response status:", response.status);

    const result = await response.json();

    if (result.error) {
      console.error("HF API error:", result.error, result.estimated_time ? `(est. wait: ${result.estimated_time}s)` : "");
      return res.status(500).json({ error: result.error, estimated_time: result.estimated_time });
    }

    // Extract text whether HF returns array [{text: '...'}] or object {text: '...'}
    const transcribedText = Array.isArray(result) ? (result[0]?.text || "") : (result.text || "");
    console.log("✓ Transcribed Konkani Text:", transcribedText);

    res.json({ text: transcribedText });
  } catch (err) {
    console.error("Server error:", err);
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`✓ Konkani Whisper UI running on http://localhost:${PORT}`));