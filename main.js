/**
 * RescueText PH frontend logic.
 */

const CONFIG = {
    API_BASE_URL: 'http://localhost:5000',
    API_TIMEOUT_MS: 30000,
    MAX_TEXT_LENGTH: 5000,
    DEFAULT_MODEL: 'transformer'
};

const EXAMPLES = [
    {
        text: 'Need rescue sa Brgy. San Isidro, baha na hanggang bubong. May bata at senior na stranded.',
        category: 'Rescue or Urgent Needs'
    },
    {
        text: 'Two injured residents near the collapsed bridge need medical assistance immediately.',
        category: 'Medical or Casualties'
    },
    {
        text: 'Evacuation center at City High School is open. Bring water, IDs, and blankets.',
        category: 'Evacuation or Displacement'
    },
    {
        text: 'Power lines are down along Mabini Street after the typhoon. Avoid the area.',
        category: 'Infrastructure Damage'
    },
    {
        text: 'Selling raincoats and flashlights at discounted prices today only.',
        category: 'Not Humanitarian'
    }
];

let appState = {
    isAnalyzing: false,
    lastResult: null
};

document.addEventListener('DOMContentLoaded', initializeApp);

function initializeApp() {
    setupExamples();
    setupEventListeners();
    loadPreferences();
    checkAPIHealth();
}

function setupEventListeners() {
    document.getElementById('textInput').addEventListener('input', (event) => {
        updateCharCounter(event.target.value.length);
        clearResults();
    });

    document.getElementById('textInput').addEventListener('keydown', (event) => {
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
            analyzePost();
        }
    });

    document.getElementById('modelSelect').addEventListener('change', () => {
        savePreferences();
        clearResults();
    });

    document.getElementById('analyzeBtn').addEventListener('click', analyzePost);
    document.getElementById('clearBtn').addEventListener('click', clearForm);
}

function setupExamples() {
    const container = document.getElementById('examples');
    container.innerHTML = EXAMPLES.map((example, index) => (
        `<button type="button" class="example-btn" data-index="${index}">
            <span>${escapeHtml(example.category)}</span>
            <strong>${escapeHtml(example.text)}</strong>
        </button>`
    )).join('');

    container.querySelectorAll('button').forEach((button) => {
        button.addEventListener('click', () => {
            const text = EXAMPLES[Number(button.dataset.index)].text;
            document.getElementById('textInput').value = text;
            updateCharCounter(text.length);
            clearResults();
        });
    });
}

async function checkAPIHealth() {
    const status = document.getElementById('apiStatus');
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/health`);
        const data = await response.json();
        if (!response.ok || !data.success) {
            throw new Error(data.error || `HTTP ${response.status}`);
        }

        status.textContent = `API ready: ${data.models.join(', ') || 'no models'}`;
        status.className = 'api-status ready';
        await loadModels();
    } catch (error) {
        status.textContent = 'API offline. Start the Flask backend on localhost:5000.';
        status.className = 'api-status offline';
    }
}

async function loadModels() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/models`);
        const data = await response.json();
        if (!data.success) return;

        const modelSelect = document.getElementById('modelSelect');
        const currentValue = modelSelect.value;
        const models = Object.entries(data.models);
        if (models.length) {
            modelSelect.innerHTML = models.map(([id, info]) => (
                `<option value="${id}">${escapeHtml(info.name || id)}</option>`
            )).join('');
            modelSelect.value = models.some(([id]) => id === currentValue) ? currentValue : data.default_model;
        }
    } catch (error) {
        console.warn('Could not load models:', error.message);
    }
}

async function analyzePost() {
    const text = document.getElementById('textInput').value.trim();
    const model = document.getElementById('modelSelect').value || CONFIG.DEFAULT_MODEL;

    if (!text) {
        showError('Enter a disaster-related post to analyze.');
        return;
    }

    if (text.length > CONFIG.MAX_TEXT_LENGTH) {
        showError(`Text exceeds ${CONFIG.MAX_TEXT_LENGTH} characters.`);
        return;
    }

    if (appState.isAnalyzing) return;

    setAnalyzing(true);
    clearResults();

    try {
        const result = await predictText(text, model);
        if (!result.success) {
            showError(result.error || 'Prediction failed.');
            return;
        }
        appState.lastResult = result;
        displayResult(result);
        savePreferences();
    } catch (error) {
        showError(error.message);
    } finally {
        setAnalyzing(false);
    }
}

async function predictText(text, model) {
    const request = fetch(`${CONFIG.API_BASE_URL}/api/predict`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text, model})
    });

    const timeout = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Request timeout. The model may still be loading.')), CONFIG.API_TIMEOUT_MS);
    });

    const response = await Promise.race([request, timeout]);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(data.error || data.message || `HTTP ${response.status}`);
    }
    return data;
}

function displayResult(result) {
    const container = document.getElementById('resultContainer');
    const urgency = result.urgency || 'low';
    container.className = `result-container active urgency-${urgency}`;

    document.getElementById('categoryText').textContent = result.category_display || formatLabel(result.category);
    document.getElementById('urgencyText').textContent = result.urgency_display || formatLabel(urgency);
    document.getElementById('confidenceScore').textContent = `${(result.confidence * 100).toFixed(1)}%`;
    document.getElementById('modelUsed').textContent = formatLabel(result.model);
    const actionability = result.actionability || {};
    const actionabilityConfidence = typeof actionability.confidence === 'number'
        ? ` ${(actionability.confidence * 100).toFixed(1)}%`
        : '';
    document.getElementById('actionabilityText').textContent = `${actionability.display_name || 'Pending'}${actionabilityConfidence}`;
    document.getElementById('inferenceTime').textContent = `${result.inference_time_ms.toFixed(2)} ms`;

    const topList = document.getElementById('topPredictions');
    topList.innerHTML = (result.top_predictions || []).map((item) => (
        `<li><span>${escapeHtml(item.display_name || formatLabel(item.category))}</span><strong>${(item.confidence * 100).toFixed(1)}%</strong></li>`
    )).join('');

    const preprocessing = result.preprocessing || {};
    document.getElementById('cleanedText').textContent = preprocessing.cleaned_text || '';
    document.getElementById('tokensText').textContent = (preprocessing.tokens || []).join(', ');

    container.scrollIntoView({behavior: 'smooth', block: 'nearest'});
}

function setAnalyzing(isAnalyzing) {
    appState.isAnalyzing = isAnalyzing;
    document.getElementById('analyzeBtn').disabled = isAnalyzing;
    document.getElementById('clearBtn').disabled = isAnalyzing;
    document.getElementById('textInput').disabled = isAnalyzing;
    document.getElementById('modelSelect').disabled = isAnalyzing;
    document.getElementById('loading').classList.toggle('active', isAnalyzing);
}

function clearForm() {
    document.getElementById('textInput').value = '';
    updateCharCounter(0);
    clearResults();
    document.getElementById('textInput').focus();
}

function clearResults() {
    document.getElementById('resultContainer').className = 'result-container';
    clearError();
}

function showError(message) {
    const element = document.getElementById('errorMessage');
    element.textContent = message;
    element.classList.add('active');
}

function clearError() {
    document.getElementById('errorMessage').classList.remove('active');
}

function updateCharCounter(count) {
    document.getElementById('charCount').textContent = count;
}

function savePreferences() {
    try {
        localStorage.setItem('rescuetext_preferences', JSON.stringify({
            model: document.getElementById('modelSelect').value
        }));
    } catch (error) {
        console.warn('Could not save preferences:', error.message);
    }
}

function loadPreferences() {
    try {
        const stored = JSON.parse(localStorage.getItem('rescuetext_preferences') || '{}');
        if (stored.model) {
            document.getElementById('modelSelect').value = stored.model;
        }
    } catch (error) {
        console.warn('Could not load preferences:', error.message);
    }
}

function formatLabel(value) {
    return String(value || '').replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {analyzePost, clearForm, predictText, displayResult};
}
