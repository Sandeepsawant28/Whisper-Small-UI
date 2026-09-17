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
  const noiseLockBadge = document.getElementById('noise-lock-badge');

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

  // Vault Modal & Header Menu Elements
  const btnOpenVault = document.getElementById('btn-open-vault');
  const btnCloseVault = document.getElementById('btn-close-vault');
  const vaultModalBackdrop = document.getElementById('vault-modal-backdrop');
  const headerVaultBadge = document.getElementById('header-vault-badge');

  // Background Noise Level Monitor Elements
  const noiseMonitorCard = document.getElementById('noise-monitor-card');
  const noiseBadge = document.getElementById('noise-badge');
  const noiseMeterFill = document.getElementById('noise-meter-fill');
  const noiseDbText = document.getElementById('noise-db-text');
  const noiseTrendBox = document.getElementById('noise-trend-box');
  const noiseTrendText = document.getElementById('noise-trend-text');
  const noiseAlertBanner = document.getElementById('noise-alert-banner');

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
  let chunkSliceInterval = null;

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

  // --- REAL-TIME AUDIO MONITOR & BACKGROUND NOISE BLOCKER ---
  let isNoiseBlocked = false;
  let noiseHistory = [];
  let ambientNoiseFloor = 8;
  let consecutiveIncreasingFrames = 0;
  let consecutiveHighNoiseFrames = 0;
  let lastSmoothedLevel = 0;

  function initAudioAnalysis(stream) {
    if (audioCtx && analyser) return;
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

  function updateNoiseMonitor(currentRawAvg) {
    if (!noiseMeterFill || !noiseBadge) return;

    if (!analyser) {
      noiseMeterFill.style.width = '6%';
      if (noiseDbText) noiseDbText.textContent = '0 dB (Ready)';
      if (noiseBadge) {
        noiseBadge.className = 'noise-badge low';
        noiseBadge.textContent = 'Quiet (Optimal)';
      }
      if (noiseTrendText) noiseTrendText.textContent = 'Environment quiet • Ready to monitor';
      if (noiseAlertBanner) noiseAlertBanner.style.display = 'none';
      if (noiseMonitorCard) {
        noiseMonitorCard.classList.remove('high-alert', 'elevated');
      }
      return;
    }

    // Smooth the raw input level
    const smoothed = (lastSmoothedLevel * 0.65) + (currentRawAvg * 0.35);
    lastSmoothedLevel = smoothed;

    // Track rolling history (last 24 frames)
    noiseHistory.push(smoothed);
    if (noiseHistory.length > 24) noiseHistory.shift();

    // Moving average of recent audio volume
    const historySum = noiseHistory.reduce((a, b) => a + b, 0);
    const movingAvg = historySum / noiseHistory.length;

    // Adaptive noise floor tracking
    if (smoothed < ambientNoiseFloor) {
      ambientNoiseFloor = smoothed;
    } else {
      ambientNoiseFloor = (ambientNoiseFloor * 0.995) + (smoothed * 0.005);
    }

    // Normalized meter fill 0% to 100%
    const meterPercent = Math.min(100, Math.max(6, Math.round((smoothed / 135) * 100)));
    noiseMeterFill.style.width = `${meterPercent}%`;

    // Estimated ambient decibels (20 dB quiet room to 95 dB loud noise/voices)
    const estimatedDb = Math.min(95, Math.max(20, Math.round(20 + (smoothed / 160) * 75)));
    if (noiseDbText) noiseDbText.textContent = `${estimatedDb} dB`;

    // Detect if background noise or voice is increasing
    const isRising = smoothed > (movingAvg * 1.25) && smoothed > 30;
    if (isRising) {
      consecutiveIncreasingFrames++;
    } else {
      consecutiveIncreasingFrames = Math.max(0, consecutiveIncreasingFrames - 1);
    }
    const isNoiseIncreasing = consecutiveIncreasingFrames >= 3;

    // Excessive noise threshold (68 dB / 68% meter): Do NOT allow recording
    const isExcessiveNoise = meterPercent >= 68 || estimatedDb >= 68;

    if (isExcessiveNoise) {
      consecutiveHighNoiseFrames++;
    } else {
      consecutiveHighNoiseFrames = Math.max(0, consecutiveHighNoiseFrames - 1);
    }

    if (consecutiveHighNoiseFrames >= 3) {
      // --- RECORDING IS BLOCKED DUE TO HIGH BACKGROUND NOISE ---
      isNoiseBlocked = true;
      if (btnToggleMic) btnToggleMic.classList.add('noise-blocked');
      if (noiseLockBadge) noiseLockBadge.style.display = 'inline-flex';
      if (deckStatusBadge && !isRecording) deckStatusBadge.textContent = 'NOISE BLOCKED';
      if (recordHint && !isRecording) recordHint.textContent = 'Speech recording blocked • High background noise';

      noiseBadge.className = 'noise-badge high';
      noiseBadge.textContent = 'Blocked (Too Loud)';
      if (noiseAlertBanner) noiseAlertBanner.style.display = 'flex';
      if (noiseMonitorCard) {
        noiseMonitorCard.classList.add('high-alert');
        noiseMonitorCard.classList.remove('elevated');
      }
      if (noiseTrendText) noiseTrendText.textContent = '🚨 Speech blocked: Ambient noise exceeds 68 dB';

      // If user was actively recording and high noise persists for ~1.5s, auto-stop to prevent garbage transcription
      if (isRecording && consecutiveHighNoiseFrames >= 12) {
        stopRecording(true);
        showToast('⚠️ Recording stopped: Background noise increased too much for speech recognition.', 'error');
      }
    } else if (isNoiseBlocked && consecutiveHighNoiseFrames === 0) {
      // --- NOISE DROPPED: UNBLOCK RECORDING ---
      isNoiseBlocked = false;
      if (btnToggleMic) btnToggleMic.classList.remove('noise-blocked');
      if (noiseLockBadge) noiseLockBadge.style.display = 'none';
      if (deckStatusBadge && !isRecording) deckStatusBadge.textContent = 'MIC READY';
      if (recordHint && !isRecording) recordHint.textContent = 'Click microphone to start recording • Speak in Konkani';
      if (noiseAlertBanner) noiseAlertBanner.style.display = 'none';
      if (noiseMonitorCard) noiseMonitorCard.classList.remove('high-alert');
    } else if (isNoiseIncreasing || meterPercent >= 46 || estimatedDb >= 50) {
      // Moderate / increasing noise warning
      noiseBadge.className = 'noise-badge medium';
      noiseBadge.textContent = isNoiseIncreasing ? '⚠️ Noise Increasing' : 'Moderate Noise';
      if (noiseAlertBanner && !isNoiseBlocked) noiseAlertBanner.style.display = isNoiseIncreasing ? 'flex' : 'none';
      if (noiseMonitorCard) {
        noiseMonitorCard.classList.remove('high-alert');
        noiseMonitorCard.classList.add('elevated');
      }
      if (noiseTrendText) {
        noiseTrendText.textContent = isNoiseIncreasing 
          ? '⚠️ Background voice or noise is rising!' 
          : 'Moderate ambient background sound';
      }
    } else {
      // Quiet / Optimal for Whisper Speech Recognition
      noiseBadge.className = 'noise-badge low';
      noiseBadge.textContent = 'Quiet (Optimal)';
      if (noiseAlertBanner && !isNoiseBlocked) noiseAlertBanner.style.display = 'none';
      if (noiseMonitorCard && !isNoiseBlocked) {
        noiseMonitorCard.classList.remove('high-alert', 'elevated');
      }
      if (noiseTrendText) noiseTrendText.textContent = 'Environment quiet & optimal for Whisper';
    }
  }

  function monitorAudioStream() {
    if (!analyser) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(dataArray);

    // VU Meter Level
    let sum = 0;
    for (let i = 0; i < bufferLength; i++) sum += dataArray[i];
    let average = sum / bufferLength;
    let vuWidth = Math.min(100, Math.max(0, (average / 160) * 100));
    if (vuMeterFill) vuMeterFill.style.width = `${vuWidth}%`;

    // Update Real-Time Background Noise Detector & Enforcement
    updateNoiseMonitor(average);

    animFrameId = requestAnimationFrame(monitorAudioStream);
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
    if (isNoiseBlocked) {
      showToast('⚠️ Cannot record speech: Background noise is too high (68+ dB). Move to a quieter area.', 'error');
      if (noiseAlertBanner) {
        noiseAlertBanner.style.display = 'flex';
      }
      return;
    }

    try {
      if (!micStream) {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      }
    } catch (err) {
      showToast('Microphone access denied or unavailable in browser.', 'error');
      console.error(err);
      return;
    }

    initAudioAnalysis(micStream);
    if (!animFrameId) {
      monitorAudioStream();
    }

    currentSessionNum = (vaultRecordings.filter(r => r.type === 'mic').length || 0) + 1;
    currentSessionId = 'rec_' + Date.now();
    currentSessionChunks = [];
    currentSessionTexts = [];
    audioChunksQueue = [];

    // Show live recording banner in vault
    if (vaultLiveBanner) vaultLiveBanner.style.display = 'block';
    if (vaultLiveSessionNum) vaultLiveSessionNum.textContent = `#${currentSessionNum}`;
    if (vaultLiveTimer) vaultLiveTimer.textContent = '00:00';

    // FIX: isRecording must be set to true BEFORE startNextSlice() is called below.
    // Previously this was set further down (after the setInterval was created),
    // which meant the very first call to startNextSlice() saw isRecording === false,
    // its guard clause returned immediately, and mediaRecorder was NEVER created.
    // That silently broke the entire pipeline: no audio was ever captured or sent
    // to /transcribe, even though the server and UI looked healthy.
    isRecording = true;

    function startNextSlice() {
      if (!isRecording || !micStream) return;

      try {
        mediaRecorder = new MediaRecorder(micStream);
      } catch (e) {
        console.error('Failed to create slice recorder:', e);
        return;
      }

      let sliceParts = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          sliceParts.push(e.data);
          currentSessionChunks.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        if (sliceParts.length > 0) {
          const chunkBlob = new Blob(sliceParts, { type: 'audio/webm' });
          if (chunkBlob.size > 800) {
            sendAudioChunkToWhisper(chunkBlob, currentSessionId);
          }
        }
      };

      mediaRecorder.start();
    }

    startNextSlice();

    // Slices audio every CHUNK_INTERVAL_MS (4s) into clean, standalone WebM files with complete headers
    clearInterval(chunkSliceInterval);
    chunkSliceInterval = setInterval(() => {
      if (!isRecording) return;
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        startNextSlice();
      }
    }, CHUNK_INTERVAL_MS);

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
    showToast('✓ Spoken recording saved to Vault! Ready to listen & download', 'success');
  }

  function stopRecording(stoppedDueToNoise = false) {
    isRecording = false;

    if (vaultLiveBanner) vaultLiveBanner.style.display = 'none';
    if (btnToggleMic) btnToggleMic.classList.remove('recording');
    if (micBtnWrapper) micBtnWrapper.classList.remove('active');
    if (liveRecDot) liveRecDot.classList.remove('active');

    if (stoppedDueToNoise) {
      if (deckStatusBadge) deckStatusBadge.textContent = 'NOISE BLOCKED';
      if (recordHint) recordHint.textContent = 'Recording stopped • Background noise too high';
    } else {
      if (deckStatusBadge) deckStatusBadge.textContent = isNoiseBlocked ? 'NOISE BLOCKED' : 'MIC PAUSED';
      if (recordHint) recordHint.textContent = isNoiseBlocked ? 'Speech blocked • Environment too noisy' : 'Recording stopped • Click microphone to record again';
    }

    clearInterval(recTimerInterval);
    clearInterval(chunkSliceInterval);

    if (mediaRecorder && mediaRecorder.state === 'recording') {
      try { mediaRecorder.stop(); } catch (e) {}
    }

    setTimeout(() => {
      finalizeMicSession();
    }, 250);

    if (!stoppedDueToNoise) {
      showToast('⏹️ Recording Stopped', 'info');
    }
  }

  // Single Record Button Toggle Listener
  if (btnToggleMic) {
    btnToggleMic.addEventListener('click', () => {
      if (!isRecording) {
        if (isNoiseBlocked) {
          showToast('⚠️ Cannot record speech: Excessive background noise detected! Please quiet your environment.', 'error');
          if (noiseAlertBanner) {
            noiseAlertBanner.style.display = 'flex';
          }
          return;
        }
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
    if (headerVaultBadge) {
      headerVaultBadge.textContent = vaultRecordings.length;
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

  // --- VAULT POP-UP MODAL CONTROLLERS ---
  function openVaultModal() {
    if (!vaultModalBackdrop) return;
    vaultModalBackdrop.style.display = 'flex';
    document.body.style.overflow = 'hidden';
    renderVaultCards(vaultSearchInput ? vaultSearchInput.value : '');
    if (window.lucide) {
      lucide.createIcons();
    }
    if (vaultSearchInput) {
      setTimeout(() => vaultSearchInput.focus(), 150);
    }
  }

  function closeVaultModal() {
    if (!vaultModalBackdrop) return;
    vaultModalBackdrop.style.display = 'none';
    document.body.style.overflow = '';
  }

  if (btnOpenVault) {
    btnOpenVault.addEventListener('click', openVaultModal);
  }

  if (btnCloseVault) {
    btnCloseVault.addEventListener('click', closeVaultModal);
  }

  if (vaultModalBackdrop) {
    vaultModalBackdrop.addEventListener('click', (e) => {
      // Close only if backdrop itself was clicked, not modal content
      if (e.target === vaultModalBackdrop) {
        closeVaultModal();
      }
    });
  }

  // Close modal on Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && vaultModalBackdrop && vaultModalBackdrop.style.display === 'flex') {
      closeVaultModal();
    }
  });

  // Warm up ambient background noise monitoring on first user interaction
  async function initAmbientMic() {
    if (micStream) return;
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      initAudioAnalysis(micStream);
      if (!animFrameId) {
        monitorAudioStream();
      }
    } catch (e) {
      // Will be prompted when user clicks record
    }
  }

  document.addEventListener('click', function onUserClick() {
    if (!micStream) {
      initAmbientMic();
    }
    document.removeEventListener('click', onUserClick);
  }, { once: true });

  // Initialize Vault from IndexedDB on page startup
  loadVaultFromDB().then((saved) => {
    vaultRecordings = saved || [];
    renderVaultCards();
  });
});