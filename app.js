/**
 * Konkani Voice AI Studio - Frontend Controller
 * Communicates with Whisper Small LoRA server at http://localhost:5000/transcribe
 */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Lucide Icons
  if (window.lucide) {
    lucide.createIcons();
  }

  const WHISPER_API_URL = 'http://localhost:5000/transcribe';
  const WHISPER_HEALTH_URL = 'http://localhost:5000/health';
  const CHUNK_INTERVAL_MS = 4000; // Slice audio every 4 seconds for continuous live transcription

  // DOM Elements
  const btnToggleMic = document.getElementById('btn-toggle-mic');
  const micIcon = document.getElementById('mic-icon');
  const micBtnWrapper = document.getElementById('mic-btn-wrapper');
  const recTimer = document.getElementById('rec-timer');
  const recordHint = document.getElementById('record-hint');
  const deckStatusBadge = document.getElementById('deck-status-badge');
  const liveRecDot = document.getElementById('live-rec-dot');
  const vuMeterFill = document.getElementById('vu-meter-fill');
  const waveformCanvas = document.getElementById('waveformCanvas');

  // File Upload Elements
  const fileDropZone = document.getElementById('file-drop-zone');
  const audioFileInput = document.getElementById('audio-file-input');
  const dropZoneTrigger = document.getElementById('drop-zone-trigger');
  const uploadedFilePreview = document.getElementById('uploaded-file-preview');
  const previewFileName = document.getElementById('preview-file-name');
  const previewAudioPlayer = document.getElementById('preview-audio-player');
  const btnRemoveFile = document.getElementById('btn-remove-file');
  const btnTranscribeFile = document.getElementById('btn-transcribe-file');
  const transcribeBtnLabel = document.getElementById('transcribe-btn-label');

  const serverStatusPill = document.getElementById('server-status-pill');
  const serverStatusDot = document.getElementById('server-status-dot');
  const serverStatusText = document.getElementById('server-status-text');
  const btnCheckServer = document.getElementById('btn-check-server');

  const transcriptScrollBox = document.getElementById('transcript-scroll-box');
  const transcriptPlaceholder = document.getElementById('transcript-placeholder');
  const transcriptFeed = document.getElementById('transcript-feed');
  const ribbonTokensContainer = document.getElementById('ribbon-tokens-container');
  const typingCursor = document.getElementById('typing-cursor');

  const statWords = document.getElementById('stat-words');
  const statChars = document.getElementById('stat-chars');
  const statChunks = document.getElementById('stat-chunks');

  const btnCopyText = document.getElementById('btn-copy-text');
  const btnDownloadTxt = document.getElementById('btn-download-txt');
  const btnClearTranscript = document.getElementById('btn-clear-transcript');

  // Spoken Vault & Download Elements
  const vaultCountBadge = document.getElementById('vault-count-badge');
  const vaultSearchInput = document.getElementById('vault-search-input');
  const vaultSearchClear = document.getElementById('vault-search-clear');
  const btnExportAllTranscripts = document.getElementById('btn-export-all-transcripts');
  const btnClearVault = document.getElementById('btn-clear-vault');
  const vaultLiveBanner = document.getElementById('vault-live-banner');
  const vaultLiveSessionNum = document.getElementById('vault-live-session-num');
  const vaultLiveTimer = document.getElementById('vault-live-timer');
  const vaultStatTotal = document.getElementById('vault-stat-total');
  const vaultStatDuration = document.getElementById('vault-stat-duration');
  const vaultStatWords = document.getElementById('vault-stat-words');
  const vaultEmptyState = document.getElementById('vault-empty-state');
  const vaultCardsGrid = document.getElementById('vault-cards-grid');

  const toast = document.getElementById('toast');
  const toastText = document.getElementById('toast-text');
  const toastIcon = document.getElementById('toast-icon');

  // State
  let isRecording = false;
  let micStream = null;
  let mediaRecorder = null;
  let audioChunksQueue = [];
  let recSeconds = 0;
  let recTimerInterval = null;

  let audioCtx = null;
  let analyser = null;
  let animFrameId = null;

  let accumulatedText = [];
  let totalChunksTranscribed = 0;

  // Vault State
  let vaultRecordings = [];
  let currentSessionId = null;
  let currentSessionNum = 0;
  let currentSessionChunks = [];
  let currentSessionTexts = [];

  // --- TOAST NOTIFICATIONS ---
  let toastTimeout = null;
  function showToast(message, type = 'info') {
    if (!toast || !toastText) return;
    toastText.textContent = message;
    toast.className = `toast-popup show ${type}`;

    if (toastTimeout) clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => {
      toast.classList.remove('show');
    }, 3500);
  }

  // --- HEALTH CHECK PING ---
  async function checkServerHealth() {
    if (serverStatusText) serverStatusText.textContent = 'Checking Model...';
    try {
      const res = await fetch(WHISPER_HEALTH_URL, { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (data.status === 'online') {
        serverStatusPill.className = 'status-pill';
        serverStatusText.textContent = `Whisper Online (${data.model || 'Konkani LoRA'})`;
        showToast('✓ Konkani Whisper Model Connected & Ready', 'success');
      } else {
        serverStatusPill.className = 'status-pill warning';
        serverStatusText.textContent = 'Model Loading in Terminal...';
        showToast('Whisper server online, model still loading weights.', 'warning');
      }
    } catch (err) {
      serverStatusPill.className = 'status-pill offline';
      serverStatusText.textContent = 'Whisper Server Offline (Port 5000)';
      showToast('⚠️ Whisper server offline. Run "python whisper_server.py" in terminal.', 'error');
    }
  }

  if (btnCheckServer) {
    btnCheckServer.addEventListener('click', checkServerHealth);
  }
  // Initial check on page load
  checkServerHealth();

  // --- AUDIO FFT VISUALIZER & VU METER ---
  let canvasCtx = waveformCanvas ? waveformCanvas.getContext('2d') : null;

  function initAudioAnalysis(stream) {
    try {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AudioContextClass();
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 64;

      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);
    } catch (e) {
      console.warn('AudioContext initialisation fallback:', e);
    }
  }

  function drawWaveform() {
    if (!waveformCanvas || !canvasCtx) return;

    const width = waveformCanvas.width = waveformCanvas.parentElement.clientWidth;
    const height = waveformCanvas.height = waveformCanvas.parentElement.clientHeight;

    canvasCtx.clearRect(0, 0, width, height);

    const bufferLength = analyser ? analyser.frequencyBinCount : 32;
    const dataArray = new Uint8Array(bufferLength);

    if (analyser && isRecording) {
      analyser.getByteFrequencyData(dataArray);
    }

    // VU Meter Level
    let sum = 0;
    for (let i = 0; i < bufferLength; i++) sum += dataArray[i];
    let average = isRecording ? sum / bufferLength : 0;
    let vuWidth = Math.min(100, Math.max(0, (average / 160) * 100));
    if (vuMeterFill) vuMeterFill.style.width = `${vuWidth}%`;

    // Draw Spectrum Bars
    const barWidth = (width / bufferLength) * 1.5;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      let barHeight = isRecording ? (dataArray[i] / 255) * height * 0.85 : 3;
      if (barHeight < 3) barHeight = 3;

      const gradient = canvasCtx.createLinearGradient(0, height - barHeight, 0, height);
      gradient.addColorStop(0, '#00FF9D');
      gradient.addColorStop(0.5, '#34D399');
      gradient.addColorStop(1, '#10B981');

      canvasCtx.fillStyle = gradient;
      canvasCtx.fillRect(x, height - barHeight, barWidth - 2, barHeight);

      x += barWidth + 2;
    }

    if (isRecording) {
      animFrameId = requestAnimationFrame(drawWaveform);
    } else {
      // Idle line
      canvasCtx.fillStyle = 'rgba(16, 185, 129, 0.2)';
      canvasCtx.fillRect(0, height / 2 - 1, width, 2);
    }
  }

  // Draw initial idle visualizer
  if (waveformCanvas) {
    drawWaveform();
  }

  // --- RECORDING TIMER ---
  function updateTimer() {
    const mins = Math.floor(recSeconds / 60).toString().padStart(2, '0');
    const secs = (recSeconds % 60).toString().padStart(2, '0');
    const timeStr = `${mins}:${secs}`;
    if (recTimer) recTimer.textContent = timeStr;
    if (vaultLiveTimer) vaultLiveTimer.textContent = timeStr;
  }

  // --- SEND AUDIO CHUNK TO WHISPER SERVER ---
  async function sendAudioChunkToWhisper(blob, sessionId = null) {
    if (!blob || blob.size === 0) return;

    const formData = new FormData();
    formData.append('audio', blob, 'audio_chunk.webm');

    try {
      const response = await fetch(WHISPER_API_URL, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || `HTTP ${response.status}`);
      }

      const data = await response.json();
      if (data && data.text && data.text.trim().length > 0) {
        const trimmed = data.text.trim();
        handleReceivedTranscription(trimmed);

        // Associate with live recording session or update finished session card
        if (sessionId) {
          currentSessionTexts.push(trimmed);
          const existingItem = vaultRecordings.find(r => r.id === sessionId);
          if (existingItem) {
            existingItem.transcription = currentSessionTexts.join(' ').trim();
            saveVaultItemToDB(existingItem);
            renderVaultCards(vaultSearchInput ? vaultSearchInput.value : '');
          }
        }
      }
    } catch (err) {
      console.warn('Transcription request error:', err);
      showToast(`Transcription warning: ${err.message}`, 'error');
    }
  }

  // --- HANDLE INCOMING TRANSCRIPTION TEXT ---
  function handleReceivedTranscription(konkaniSentence, sourceLabel = null) {
    if (!konkaniSentence || konkaniSentence.trim().length === 0) return;

    totalChunksTranscribed++;
    accumulatedText.push(konkaniSentence);

    // Hide placeholder, reveal feed
    if (transcriptPlaceholder) transcriptPlaceholder.style.display = 'none';
    if (transcriptFeed) transcriptFeed.style.display = 'flex';

    // Create entry element
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const entryEl = document.createElement('div');
    entryEl.className = 'transcript-entry';
    const tagText = sourceLabel ? `📁 ${sourceLabel}` : `Chunk #${totalChunksTranscribed}`;
    entryEl.innerHTML = `
      <div class="entry-meta">
        <span>${tagText} • ${timestamp}</span>
        <span style="color:var(--emerald-400); font-weight:700;">WHISPER KONKANI</span>
      </div>
      <div class="entry-text">${konkaniSentence}</div>
    `;

    transcriptFeed.appendChild(entryEl);

    // Auto-scroll to latest
    if (transcriptScrollBox) {
      transcriptScrollBox.scrollTop = transcriptScrollBox.scrollHeight;
    }

    // Update Word Ribbon Tokens
    updateWordRibbon(konkaniSentence);

    // Update Statistics
    updateStatistics();
  }

  // --- UPDATE WORD RIBBON WITH TOKENS ---
  function updateWordRibbon(text) {
    if (!ribbonTokensContainer) return;
    const words = text.split(/\s+/).filter(w => w.trim().length > 0);

    let html = '';
    words.forEach(w => {
      html += `<span class="token-chip">[${w}]</span>`;
    });

    ribbonTokensContainer.innerHTML = html;
  }

  // --- UPDATE STATISTICS ---
  function updateStatistics() {
    const fullText = accumulatedText.join(' ').trim();
    const words = fullText.length > 0 ? fullText.split(/\s+/).filter(w => w.length > 0).length : 0;
    const chars = fullText.length;

    if (statWords) statWords.textContent = words;
    if (statChars) statChars.textContent = chars;
    if (statChunks) statChunks.textContent = totalChunksTranscribed;
  }

  // --- RECORDING CONTROLS ---
  async function startRecording() {
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      showToast('Microphone access denied or unavailable in browser.', 'error');
      console.error(err);
      return;
    }

    initAudioAnalysis(micStream);

    currentSessionNum = (vaultRecordings.filter(r => r.type === 'mic').length || 0) + 1;
    currentSessionId = 'rec_' + Date.now();
    currentSessionChunks = [];
    currentSessionTexts = [];
    audioChunksQueue = [];

    // Show live recording banner in vault
    if (vaultLiveBanner) vaultLiveBanner.style.display = 'block';
    if (vaultLiveSessionNum) vaultLiveSessionNum.textContent = `#${currentSessionNum}`;
    if (vaultLiveTimer) vaultLiveTimer.textContent = '00:00';

    mediaRecorder = new MediaRecorder(micStream);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        audioChunksQueue.push(e.data);
        currentSessionChunks.push(e.data);
        const chunkBlob = new Blob([e.data], { type: 'audio/webm' });
        sendAudioChunkToWhisper(chunkBlob, currentSessionId);
      }
    };

    mediaRecorder.onstop = () => {
      finalizeMicSession();
    };

    // Slices audio every CHUNK_INTERVAL_MS (4s) for continuous live transcription
    mediaRecorder.start(CHUNK_INTERVAL_MS);

    isRecording = true;
    btnToggleMic.classList.add('recording');
    if (micBtnWrapper) micBtnWrapper.classList.add('active');
    if (liveRecDot) liveRecDot.classList.add('active');
    if (deckStatusBadge) deckStatusBadge.textContent = 'RECORDING LIVE';
    if (recordHint) recordHint.textContent = 'Listening continuously... Speak in Konkani!';

    // Reset and start timer
    recSeconds = 0;
    updateTimer();
    clearInterval(recTimerInterval);
    recTimerInterval = setInterval(() => {
      recSeconds++;
      updateTimer();
    }, 1000);

    // Start FFT waveform loop
    drawWaveform();
    showToast('🎙️ Live Recording Started • Speaking Konkani', 'info');
  }

  function finalizeMicSession() {
    if (currentSessionChunks.length === 0) return;

    const fullBlob = new Blob(currentSessionChunks, { type: 'audio/webm' });
    if (fullBlob.size === 0) return;

    const sessionRecord = {
      id: currentSessionId,
      type: 'mic',
      title: `Voice Session #${currentSessionNum}`,
      timestamp: new Date().toISOString(),
      displayDate: new Date().toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' }),
      displayTime: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      durationSeconds: recSeconds,
      audioBlob: fullBlob,
      audioUrl: URL.createObjectURL(fullBlob),
      transcription: currentSessionTexts.join(' ').trim(),
      sizeBytes: fullBlob.size
    };

    addRecordToVault(sessionRecord);
    showToast('✓ Spoken recording saved to Vault below! Ready to listen & download', 'success');
  }

  function stopRecording() {
    isRecording = false;

    if (vaultLiveBanner) vaultLiveBanner.style.display = 'none';
    if (btnToggleMic) btnToggleMic.classList.remove('recording');
    if (micBtnWrapper) micBtnWrapper.classList.remove('active');
    if (liveRecDot) liveRecDot.classList.remove('active');
    if (deckStatusBadge) deckStatusBadge.textContent = 'MIC PAUSED';
    if (recordHint) recordHint.textContent = 'Recording stopped • Click microphone to record again';

    clearInterval(recTimerInterval);

    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      try { mediaRecorder.stop(); } catch (e) {}
    }

    if (micStream) {
      micStream.getTracks().forEach(t => t.stop());
    }

    if (animFrameId) cancelAnimationFrame(animFrameId);
    drawWaveform();
    showToast('⏹️ Recording Stopped', 'info');
  }

  // Single Record Button Toggle Listener
  if (btnToggleMic) {
    btnToggleMic.addEventListener('click', () => {
      if (!isRecording) {
        startRecording();
      } else {
        stopRecording();
      }
    });
  }

  // --- ACTION TOOLBAR LISTENERS ---

  // Copy All Text
  if (btnCopyText) {
    btnCopyText.addEventListener('click', async () => {
      const fullText = accumulatedText.join('\n\n').trim();
      if (!fullText) {
        showToast('No transcribed text to copy yet.', 'warning');
        return;
      }
      try {
        await navigator.clipboard.writeText(fullText);
        showToast('✓ Konkani text copied to clipboard!', 'success');
      } catch (e) {
        showToast('Could not copy to clipboard.', 'error');
      }
    });
  }

  // Download .txt
  if (btnDownloadTxt) {
    btnDownloadTxt.addEventListener('click', () => {
      const fullText = accumulatedText.join('\n\n').trim();
      if (!fullText) {
        showToast('No transcribed text to save yet.', 'warning');
        return;
      }
      const blob = new Blob([fullText], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `konkani_transcript_${Date.now()}.txt`;
      a.click();
      URL.revokeObjectURL(url);
      showToast('✓ Transcript downloaded as .txt', 'success');
    });
  }

  // Clear Transcript
  if (btnClearTranscript) {
    btnClearTranscript.addEventListener('click', () => {
      if (accumulatedText.length === 0) return;
      if (confirm('Clear all transcribed text?')) {
        accumulatedText = [];
        totalChunksTranscribed = 0;
        if (transcriptFeed) {
          transcriptFeed.innerHTML = '';
          transcriptFeed.style.display = 'none';
        }
        if (transcriptPlaceholder) transcriptPlaceholder.style.display = 'flex';
        if (ribbonTokensContainer) {
          ribbonTokensContainer.innerHTML = '<span class="token-chip placeholder">Tokens will appear here as you speak...</span>';
        }
        updateStatistics();
        showToast('🗑️ Transcript cleared', 'info');
      }
    });
  }

  // --- AUDIO FILE UPLOAD & TRANSCRIPTION ---
  let selectedAudioFile = null;
  let isTranscribingFile = false;

  function handleFileSelected(file) {
    if (!file) return;

    selectedAudioFile = file;
    const sizeKb = (file.size / 1024).toFixed(1);
    if (previewFileName) previewFileName.textContent = `${file.name} (${sizeKb} KB)`;

    // Set audio player source
    if (previewAudioPlayer) {
      previewAudioPlayer.src = URL.createObjectURL(file);
    }

    // Toggle view
    if (dropZoneTrigger) dropZoneTrigger.style.display = 'none';
    if (uploadedFilePreview) uploadedFilePreview.style.display = 'flex';

    showToast(`✓ Loaded ${file.name}. Ready to transcribe!`, 'info');
  }

  function resetFileUpload() {
    selectedAudioFile = null;
    isTranscribingFile = false;
    if (audioFileInput) audioFileInput.value = '';
    if (previewAudioPlayer) {
      previewAudioPlayer.pause();
      previewAudioPlayer.src = '';
    }
    if (uploadedFilePreview) uploadedFilePreview.style.display = 'none';
    if (dropZoneTrigger) dropZoneTrigger.style.display = 'flex';
    if (btnTranscribeFile) btnTranscribeFile.disabled = false;
    if (transcribeBtnLabel) transcribeBtnLabel.textContent = 'Transcribe Uploaded File';
  }

  if (dropZoneTrigger && audioFileInput) {
    dropZoneTrigger.addEventListener('click', () => audioFileInput.click());
  }

  if (audioFileInput) {
    audioFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelected(e.target.files[0]);
      }
    });
  }

  // Drag & drop handlers
  if (fileDropZone) {
    ['dragenter', 'dragover'].forEach(eventName => {
      fileDropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        fileDropZone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      fileDropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        fileDropZone.classList.remove('dragover');
      });
    });

    fileDropZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files[0]) {
        handleFileSelected(dt.files[0]);
      }
    });
  }

  if (btnRemoveFile) {
    btnRemoveFile.addEventListener('click', resetFileUpload);
  }

  // Transcribe Uploaded File
  if (btnTranscribeFile) {
    btnTranscribeFile.addEventListener('click', async () => {
      if (!selectedAudioFile || isTranscribingFile) return;

      isTranscribingFile = true;
      btnTranscribeFile.disabled = true;
      if (transcribeBtnLabel) transcribeBtnLabel.textContent = 'Transcribing with Whisper...';
      showToast(`⏳ Transcribing ${selectedAudioFile.name}...`, 'info');

      const formData = new FormData();
      formData.append('audio', selectedAudioFile, selectedAudioFile.name);

      try {
        const response = await fetch(WHISPER_API_URL, {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.error || `HTTP ${response.status}`);
        }

        const data = await response.json();
        if (data && data.text && data.text.trim().length > 0) {
          const transcribedText = data.text.trim();
          handleReceivedTranscription(transcribedText, selectedAudioFile.name);
          showToast(`✓ Successfully transcribed ${selectedAudioFile.name}!`, 'success');

          // Save uploaded audio & transcript to Vault
          const fileRecord = {
            id: 'file_' + Date.now(),
            type: 'file',
            title: selectedAudioFile.name,
            timestamp: new Date().toISOString(),
            displayDate: new Date().toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' }),
            displayTime: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            durationSeconds: Math.round((previewAudioPlayer && previewAudioPlayer.duration) ? previewAudioPlayer.duration : 0),
            audioBlob: selectedAudioFile,
            audioUrl: URL.createObjectURL(selectedAudioFile),
            transcription: transcribedText,
            sizeBytes: selectedAudioFile.size
          };
          addRecordToVault(fileRecord);
        } else {
          showToast('Transcription returned empty or silence.', 'warning');
        }
      } catch (err) {
        console.error('File transcription error:', err);
        showToast(`Transcription error: ${err.message}`, 'error');
      } finally {
        isTranscribingFile = false;
        if (btnTranscribeFile) btnTranscribeFile.disabled = false;
        if (transcribeBtnLabel) transcribeBtnLabel.textContent = 'Transcribe Uploaded File';
      }
    });
  }

  // ==========================================================================
  // SPOKEN AUDIO & TRANSCRIPTION VAULT CONTROLLER (INDEXEDDB + DOWNLOADS)
  // ==========================================================================

  const DB_NAME = 'KonkaniVoiceVaultDB';
  const DB_VERSION = 1;
  const STORE_NAME = 'recordings';

  function openDB() {
    return new Promise((resolve) => {
      if (!window.indexedDB) {
        resolve(null);
        return;
      }
      try {
        const request = indexedDB.open(DB_NAME, DB_VERSION);
        request.onupgradeneeded = (e) => {
          const db = e.target.result;
          if (!db.objectStoreNames.contains(STORE_NAME)) {
            db.createObjectStore(STORE_NAME, { keyPath: 'id' });
          }
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = (e) => {
          console.warn('IndexedDB open error:', e);
          resolve(null);
        };
      } catch (e) {
        console.warn('IndexedDB access blocked or unsupported:', e);
        resolve(null);
      }
    });
  }

  async function loadVaultFromDB() {
    try {
      const db = await openDB();
      if (!db) return [];
      return new Promise((resolve) => {
        const tx = db.transaction(STORE_NAME, 'readonly');
        const store = tx.objectStore(STORE_NAME);
        const req = store.getAll();
        req.onsuccess = () => {
          const records = req.result || [];
          records.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
          records.forEach(r => {
            if (r.audioBlob && !r.audioUrl) {
              r.audioUrl = URL.createObjectURL(r.audioBlob);
            }
          });
          resolve(records);
        };
        req.onerror = () => resolve([]);
      });
    } catch (e) {
      console.warn('loadVaultFromDB error:', e);
      return [];
    }
  }

  async function saveVaultItemToDB(item) {
    try {
      const db = await openDB();
      if (!db) return;
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const toStore = { ...item };
      delete toStore.audioUrl;
      store.put(toStore);
    } catch (e) {
      console.warn('saveVaultItemToDB error:', e);
    }
  }

  async function deleteVaultItemFromDB(id) {
    try {
      const db = await openDB();
      if (!db) return;
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      store.delete(id);
    } catch (e) {
      console.warn('deleteVaultItemFromDB error:', e);
    }
  }

  async function clearVaultFromDB() {
    try {
      const db = await openDB();
      if (!db) return;
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      store.clear();
    } catch (e) {
      console.warn('clearVaultFromDB error:', e);
    }
  }

  function formatDuration(sec) {
    if (!sec || isNaN(sec) || sec <= 0) return '00:00';
    const mins = Math.floor(sec / 60).toString().padStart(2, '0');
    const secs = (sec % 60).toString().padStart(2, '0');
    return `${mins}:${secs}`;
  }

  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 KB';
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB`;
    return `${(kb / 1024).toFixed(2)} MB`;
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/[&<>"']/g, (m) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;'
    })[m]);
  }

  function addRecordToVault(record) {
    vaultRecordings.unshift(record);
    saveVaultItemToDB(record);
    renderVaultCards(vaultSearchInput ? vaultSearchInput.value : '');
  }

  function renderVaultCards(filterQuery = '') {
    const query = filterQuery.toLowerCase().trim();
    const filtered = vaultRecordings.filter(rec => {
      if (!query) return true;
      const titleMatch = (rec.title || '').toLowerCase().includes(query);
      const textMatch = (rec.transcription || '').toLowerCase().includes(query);
      const dateMatch = (rec.displayDate || '').toLowerCase().includes(query);
      return titleMatch || textMatch || dateMatch;
    });

    // Update Counter Badge
    if (vaultCountBadge) {
      vaultCountBadge.textContent = `${vaultRecordings.length} ${vaultRecordings.length === 1 ? 'RECORDING' : 'RECORDINGS'}`;
    }

    // Update Summary Metrics
    if (vaultStatTotal) vaultStatTotal.textContent = vaultRecordings.length;

    const totalSeconds = vaultRecordings.reduce((sum, r) => sum + (r.durationSeconds || 0), 0);
    if (vaultStatDuration) vaultStatDuration.textContent = formatDuration(totalSeconds);

    const totalWords = vaultRecordings.reduce((sum, r) => {
      const txt = (r.transcription || '').trim();
      return sum + (txt ? txt.split(/\s+/).filter(w => w.length > 0).length : 0);
    }, 0);
    if (vaultStatWords) vaultStatWords.textContent = totalWords;

    // Toggle Empty State vs Grid
    if (!vaultCardsGrid || !vaultEmptyState) return;

    if (filtered.length === 0) {
      vaultEmptyState.style.display = 'flex';
      vaultCardsGrid.style.display = 'none';
      vaultCardsGrid.innerHTML = '';

      if (query) {
        vaultEmptyState.querySelector('.empty-title').textContent = 'No matching recordings found';
        vaultEmptyState.querySelector('.empty-description').textContent = `No spoken recordings matched "${filterQuery}". Try a different keyword or clear the search filter.`;
      } else {
        vaultEmptyState.querySelector('.empty-title').textContent = 'No Spoken Recordings Yet';
        vaultEmptyState.querySelector('.empty-description').textContent = 'Start speaking using the Voice Recording Deck above or upload an audio file. Every spoken session will be automatically saved here so you can listen back to your voice, read what you spoke in Konkani, and download your audio files (.webm) and transcripts (.txt)!';
      }
      return;
    }

    vaultEmptyState.style.display = 'none';
    vaultCardsGrid.style.display = 'grid';

    let cardsHtml = '';
    filtered.forEach(rec => {
      const isMic = rec.type === 'mic';
      const iconName = isMic ? 'mic' : 'file-audio';
      const typeLabel = isMic ? 'Microphone Recording' : 'Uploaded File';
      const wordsCount = rec.transcription ? rec.transcription.trim().split(/\s+/).filter(w => w.length > 0).length : 0;
      const charsCount = rec.transcription ? rec.transcription.trim().length : 0;
      const hasSpeech = rec.transcription && rec.transcription.trim().length > 0;

      cardsHtml += `
        <article class="vault-item-card" data-card-id="${rec.id}">
          <div class="vault-item-header">
            <div class="vault-item-title-col">
              <div class="vault-item-title">
                <i data-lucide="${iconName}" style="width:16px; color:${isMic ? 'var(--neon-green)' : 'var(--emerald-400)'};"></i>
                <span>${escapeHtml(rec.title)}</span>
              </div>
              <div class="vault-item-timestamp">
                <span>${escapeHtml(rec.displayDate)} • ${escapeHtml(rec.displayTime)}</span>
              </div>
            </div>

            <div class="vault-item-pills">
              <span class="vault-pill duration" title="Recording Duration">
                <i data-lucide="clock" style="width:11px;"></i> ${formatDuration(rec.durationSeconds)}
              </span>
              <span class="vault-pill" title="Audio Size">
                <i data-lucide="hard-drive" style="width:11px;"></i> ${formatBytes(rec.sizeBytes)}
              </span>
            </div>
          </div>

          <!-- Audio Playback Deck -->
          <div class="vault-audio-box">
            <audio controls preload="metadata" class="vault-audio-player" src="${rec.audioUrl}"></audio>
          </div>

          <!-- Spoken Konkani Text Display Box -->
          <div class="vault-speech-content">
            <div class="vault-speech-label">
              <span><i data-lucide="sparkles" style="width:12px; margin-right:4px;"></i> कोंकणी उलयिल्लें (Transcribed Speech)</span>
              <span style="opacity:0.75; font-family:var(--font-mono); font-size:0.65rem;">WHISPER KONKANI</span>
            </div>
            <div class="vault-speech-text ${hasSpeech ? '' : 'empty'}">
              ${hasSpeech ? escapeHtml(rec.transcription) : '(Waiting for transcription or no spoken words detected)'}
            </div>
            <div class="vault-speech-meta">
              <span><strong>${wordsCount}</strong> Words</span>
              <span><strong>${charsCount}</strong> Characters</span>
              <span>${typeLabel}</span>
            </div>
          </div>

          <!-- Action Buttons Bar: Download Audio, Download Transcript, Copy, Delete -->
          <div class="vault-item-actions">
            <button class="btn-vault-action audio-dl" data-action="download-audio" data-id="${rec.id}" title="Download your voice recording (.webm)">
              <i data-lucide="download" style="width:13px;"></i>
              <span>Download Audio</span>
            </button>
            <button class="btn-vault-action text-dl" data-action="download-transcript" data-id="${rec.id}" title="Download Konkani transcript (.txt)">
              <i data-lucide="file-text" style="width:13px;"></i>
              <span>Download Text</span>
            </button>
            <button class="btn-vault-action copy-btn" data-action="copy-text" data-id="${rec.id}" title="Copy Konkani text">
              <i data-lucide="copy" style="width:13px;"></i>
              <span>Copy</span>
            </button>
            <button class="btn-vault-action delete-btn" data-action="delete" data-id="${rec.id}" title="Delete this recording">
              <i data-lucide="trash-2" style="width:14px;"></i>
            </button>
          </div>
        </article>
      `;
    });

    vaultCardsGrid.innerHTML = cardsHtml;

    // Refresh Lucide Icons
    if (window.lucide) {
      lucide.createIcons();
    }
  }

  // --- DOWNLOAD AUDIO HANDLER ---
  function downloadVaultAudio(id) {
    const rec = vaultRecordings.find(r => r.id === id);
    if (!rec) return;

    const url = rec.audioUrl || (rec.audioBlob ? URL.createObjectURL(rec.audioBlob) : null);
    if (!url) {
      showToast('Audio file unavailable for download.', 'error');
      return;
    }

    const cleanTitle = (rec.title || 'recording').replace(/[^a-zA-Z0-9_-]/g, '_').toLowerCase();
    const filename = `konkani_${cleanTitle}_${Date.now()}.webm`;

    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    showToast(`✓ Downloaded spoken audio: ${filename}`, 'success');
  }

  // --- DOWNLOAD TRANSCRIPT HANDLER ---
  function downloadVaultTranscript(id) {
    const rec = vaultRecordings.find(r => r.id === id);
    if (!rec) return;

    const textContent = `=====================================================
KONKANI VOICE AI STUDIO - RECORDING TRANSCRIPT
=====================================================
Session Title   : ${rec.title}
Date & Time     : ${rec.displayDate} at ${rec.displayTime}
Audio Duration  : ${formatDuration(rec.durationSeconds)}
Source Type     : ${rec.type === 'mic' ? 'Microphone Voice Input' : 'Uploaded Audio File'}
File Size       : ${formatBytes(rec.sizeBytes)}
=====================================================
TRANSCRIPTION (कोंकणी - Devanagari):
=====================================================
${rec.transcription ? rec.transcription : '(No Konkani text was transcribed for this session)'}
=====================================================
Generated by Konkani Voice AI Studio (Fine-Tuned Whisper Small LoRA)
`;

    const blob = new Blob([textContent], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const cleanTitle = (rec.title || 'transcript').replace(/[^a-zA-Z0-9_-]/g, '_').toLowerCase();
    const filename = `konkani_${cleanTitle}_transcript.txt`;

    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showToast(`✓ Downloaded Konkani transcript (.txt)`, 'success');
  }

  // --- EXPORT ALL TRANSCRIPTS HANDLER ---
  function exportAllTranscripts() {
    if (vaultRecordings.length === 0) {
      showToast('No recordings in vault to export.', 'warning');
      return;
    }

    let fullExport = `======================================================================
KONKANI VOICE AI STUDIO - MASTER TRANSCRIPT EXPORT
Total Recordings: ${vaultRecordings.length}
Exported On     : ${new Date().toLocaleString()}
======================================================================\n\n`;

    vaultRecordings.forEach((rec, idx) => {
      fullExport += `----------------------------------------------------------------------
#${idx + 1} | ${rec.title}
Date & Time: ${rec.displayDate} ${rec.displayTime} | Duration: ${formatDuration(rec.durationSeconds)} | Type: ${rec.type === 'mic' ? 'Mic' : 'File'}
----------------------------------------------------------------------
${rec.transcription ? rec.transcription : '(No Konkani text)'}\n\n`;
    });

    fullExport += `======================================================================
End of Konkani Voice AI Studio Export
======================================================================`;

    const blob = new Blob([fullExport], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `konkani_all_transcripts_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showToast(`✓ Exported all ${vaultRecordings.length} transcripts to .txt`, 'success');
  }

  // --- COPY TEXT HANDLER ---
  async function copyVaultText(id) {
    const rec = vaultRecordings.find(r => r.id === id);
    if (!rec || !rec.transcription) {
      showToast('No transcribed text to copy for this recording.', 'warning');
      return;
    }
    try {
      await navigator.clipboard.writeText(rec.transcription);
      showToast('✓ Konkani text copied to clipboard!', 'success');
    } catch (e) {
      showToast('Could not copy text to clipboard.', 'error');
    }
  }

  // --- DELETE RECORDING HANDLER ---
  async function deleteVaultItem(id) {
    const index = vaultRecordings.findIndex(r => r.id === id);
    if (index === -1) return;

    const item = vaultRecordings[index];
    if (item.audioUrl) {
      try { URL.revokeObjectURL(item.audioUrl); } catch (e) {}
    }

    vaultRecordings.splice(index, 1);
    await deleteVaultItemFromDB(id);
    renderVaultCards(vaultSearchInput ? vaultSearchInput.value : '');
    showToast('🗑️ Recording removed from vault.', 'info');
  }

  // --- CLEAR VAULT HANDLER ---
  async function clearEntireVault() {
    if (vaultRecordings.length === 0) return;
    if (!confirm('Are you sure you want to clear all spoken recordings from your vault? This cannot be undone.')) {
      return;
    }

    vaultRecordings.forEach(r => {
      if (r.audioUrl) {
        try { URL.revokeObjectURL(r.audioUrl); } catch (e) {}
      }
    });

    vaultRecordings = [];
    await clearVaultFromDB();
    renderVaultCards();
    showToast('🗑️ Voice Vault cleared.', 'info');
  }

  // --- VAULT EVENT LISTENERS ---
  if (vaultCardsGrid) {
    vaultCardsGrid.addEventListener('click', (e) => {
      const button = e.target.closest('[data-action]');
      if (!button) return;

      const action = button.getAttribute('data-action');
      const id = button.getAttribute('data-id');

      if (action === 'download-audio') {
        downloadVaultAudio(id);
      } else if (action === 'download-transcript') {
        downloadVaultTranscript(id);
      } else if (action === 'copy-text') {
        copyVaultText(id);
      } else if (action === 'delete') {
        deleteVaultItem(id);
      }
    });
  }

  if (vaultSearchInput) {
    vaultSearchInput.addEventListener('input', (e) => {
      const val = e.target.value;
      if (vaultSearchClear) {
        vaultSearchClear.style.display = val ? 'block' : 'none';
      }
      renderVaultCards(val);
    });
  }

  if (vaultSearchClear) {
    vaultSearchClear.addEventListener('click', () => {
      if (vaultSearchInput) vaultSearchInput.value = '';
      vaultSearchClear.style.display = 'none';
      renderVaultCards('');
    });
  }

  if (btnExportAllTranscripts) {
    btnExportAllTranscripts.addEventListener('click', exportAllTranscripts);
  }

  if (btnClearVault) {
    btnClearVault.addEventListener('click', clearEntireVault);
  }

  // Initialize Vault from IndexedDB on page startup
  loadVaultFromDB().then((saved) => {
    vaultRecordings = saved || [];
    renderVaultCards();
  });
});
