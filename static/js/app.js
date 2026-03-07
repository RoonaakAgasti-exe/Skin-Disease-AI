const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const idleView = document.getElementById('idle-view');
const loadingView = document.getElementById('loading-view');
const resultsView = document.getElementById('results-view');
const reportBtn = document.getElementById('report-btn');

let currentScanId = null;
let predictionChart = null;

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleUpload(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleUpload(e.target.files[0]);
    }
});

async function handleUpload(file) {
    if (!file.type.startsWith('image/')) {
        alert('Invalid format. Please upload a medical image.');
        return;
    }

    idleView.classList.add('hidden-view');
    resultsView.classList.add('hidden-view');
    loadingView.classList.remove('hidden-view');

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Diagnostic Engine Failure (${response.status})`);
        }

        const data = await response.json();
        renderResults(data);
    } catch (err) {
        alert(err.message);
        loadingView.classList.add('hidden-view');
        idleView.classList.remove('hidden-view');
    }
}

function renderResults(data) {
    loadingView.classList.add('hidden-view');
    resultsView.classList.remove('hidden-view');

    currentScanId = data.scan_id;

    document.getElementById('condition-name').textContent = data.top_prediction.class;

    const riskBadge = document.getElementById('risk-badge');
    riskBadge.textContent = data.top_prediction.risk_level === 'high' ? 'High Risk' : 'Low Risk';
    riskBadge.className = `badge risk-${data.top_prediction.risk_level}`;

    document.getElementById('condition-desc').textContent = data.medical_info.description;

    const medList = document.getElementById('medicine-list');
    medList.innerHTML = '';
    data.medical_info.medicines.forEach(m => {
        const li = document.createElement('li');
        li.textContent = m;
        medList.appendChild(li);
    });

    const prevList = document.getElementById('preventive-list');
    prevList.innerHTML = '';
    data.medical_info.preventives.forEach(p => {
        const li = document.createElement('li');
        li.textContent = p;
        prevList.appendChild(li);
    });

    updateChart(data.predictions);
}

function updateChart(predictions) {
    const ctx = document.getElementById('prediction-chart').getContext('2d');

    const labels = predictions.map(p => p.class);
    const scores = predictions.map(p => p.confidence * 100);

    if (predictionChart) {
        predictionChart.destroy();
    }

    predictionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Confidence (%)',
                data: scores,
                backgroundColor: 'rgba(0, 86, 179, 0.6)',
                borderColor: 'rgba(0, 86, 179, 1)',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    grid: { display: false }
                },
                y: {
                    grid: { display: false }
                }
            }
        }
    });
}

reportBtn.addEventListener('click', () => {
    if (currentScanId) {
        window.location.href = `/report/${currentScanId}`;
    }
});