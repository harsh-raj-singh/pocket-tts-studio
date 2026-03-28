/* ============================================================
   Pocket TTS Studio — Frontend Logic
   ============================================================ */

// --- Tab switching ---
function switchTab(mode) {
    const textTab = document.getElementById('tab-text');
    const urlTab = document.getElementById('tab-url');
    const textPanel = document.getElementById('panel-text');
    const urlPanel = document.getElementById('panel-url');

    if (mode === 'text') {
        textTab.classList.add('active');
        textTab.setAttribute('aria-selected', 'true');
        urlTab.classList.remove('active');
        urlTab.setAttribute('aria-selected', 'false');
        textPanel.classList.remove('hidden');
        urlPanel.classList.add('hidden');
    } else {
        urlTab.classList.add('active');
        urlTab.setAttribute('aria-selected', 'true');
        textTab.classList.remove('active');
        textTab.setAttribute('aria-selected', 'false');
        urlPanel.classList.remove('hidden');
        textPanel.classList.add('hidden');
    }
}

// --- Character count ---
const textInput = document.getElementById('text-input');
const charCount = document.getElementById('char-count');

textInput.addEventListener('input', () => {
    charCount.textContent = textInput.value.length.toLocaleString();
});

// --- Generate ---
async function generate() {
    const btn = document.getElementById('generate-btn');
    const btnContent = btn.querySelector('.btn-content');
    const btnLoading = btn.querySelector('.btn-loading');
    const statusBar = document.getElementById('status-bar');
    const statusText = document.getElementById('status-text');
    const resultCard = document.getElementById('result-card');
    const errorCard = document.getElementById('error-card');

    // Determine input mode
    const isUrlMode = document.getElementById('tab-url').classList.contains('active');
    const voice = document.getElementById('voice-select').value;

    let payload = { voice };

    if (isUrlMode) {
        const url = document.getElementById('url-input').value.trim();
        if (!url) {
            showError('Please enter a URL.');
            return;
        }
        payload.url = url;
    } else {
        const text = textInput.value.trim();
        if (!text) {
            showError('Please enter some text.');
            return;
        }
        payload.text = text;
    }

    // UI: loading state
    btn.disabled = true;
    btnContent.classList.add('hidden');
    btnLoading.classList.remove('hidden');
    statusBar.classList.remove('hidden');
    resultCard.classList.add('hidden');
    errorCard.classList.add('hidden');

    statusText.textContent = isUrlMode
        ? 'Fetching content from URL and generating audio…'
        : 'Generating audio from text…';

    try {
        const startTime = performance.now();

        const resp = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        const wallTime = ((performance.now() - startTime) / 1000).toFixed(1);

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Generation failed.');
        }

        const data = await resp.json();
        showResult(data, wallTime);

    } catch (err) {
        showError(err.message || 'Something went wrong.');
    } finally {
        btn.disabled = false;
        btnContent.classList.remove('hidden');
        btnLoading.classList.add('hidden');
        statusBar.classList.add('hidden');
    }
}

// --- Show result ---
function showResult(data, wallTime) {
    const resultCard = document.getElementById('result-card');
    const resultMeta = document.getElementById('result-meta');
    const resultStats = document.getElementById('result-stats');
    const audioPlayer = document.getElementById('audio-player');
    const downloadBtn = document.getElementById('download-btn');

    // Duration formatting
    const durationMin = Math.floor(data.duration_seconds / 60);
    const durationSec = Math.round(data.duration_seconds % 60);
    const durationStr = durationMin > 0
        ? `${durationMin}m ${durationSec}s`
        : `${durationSec}s`;

    resultMeta.textContent = `${durationStr} of audio · ${data.num_chunks} chunk${data.num_chunks > 1 ? 's' : ''} processed`;

    // Stats
    const speed = (data.duration_seconds / data.generation_time_seconds).toFixed(1);
    resultStats.innerHTML = `
        <div class="stat">
            <span class="stat-value">${durationStr}</span>
            <span class="stat-label">Duration</span>
        </div>
        <div class="stat">
            <span class="stat-value">${data.generation_time_seconds.toFixed(1)}s</span>
            <span class="stat-label">Gen Time</span>
        </div>
        <div class="stat">
            <span class="stat-value">${speed}x</span>
            <span class="stat-label">Real-time</span>
        </div>
        <div class="stat">
            <span class="stat-value">${data.char_count.toLocaleString()}</span>
            <span class="stat-label">Characters</span>
        </div>
    `;

    // Audio
    audioPlayer.src = data.download_url;
    downloadBtn.href = data.download_url;
    downloadBtn.download = data.filename;

    resultCard.classList.remove('hidden');

    // Scroll into view
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// --- Show error ---
function showError(msg) {
    const errorCard = document.getElementById('error-card');
    const errorText = document.getElementById('error-text');
    errorText.textContent = msg;
    errorCard.classList.remove('hidden');

    // Auto-hide after 6s
    setTimeout(() => {
        errorCard.classList.add('hidden');
    }, 6000);
}

// --- Keyboard shortcut: Ctrl/Cmd + Enter to generate ---
document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        generate();
    }
});
