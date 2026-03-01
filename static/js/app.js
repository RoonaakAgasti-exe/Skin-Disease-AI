// Skin Disease AI Frontend Application
class SkinAIApp {
    constructor() {
        this.currentScanId = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkHealth();
    }

    bindEvents() {
        // File upload events
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const uploadForm = document.getElementById('upload-form');

        // Drop zone events
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', this.handleDragOver.bind(this));
        dropZone.addEventListener('dragleave', this.handleDragLeave.bind(this));
        dropZone.addEventListener('drop', this.handleDrop.bind(this));

        // File input event
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));

        // Form submission
        uploadForm.addEventListener('submit', this.handleSubmit.bind(this));

        // Action buttons
        document.getElementById("download-report").addEventListener("click", this.downloadReport.bind(this));
        // New analysis button removed as requested

        // Prevent form default submission
        uploadForm.addEventListener('submit', (e) => e.preventDefault());
    }

    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        document.getElementById('drop-zone').classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        document.getElementById('drop-zone').classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        document.getElementById('drop-zone').classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    processFile(file) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif'];
        if (!validTypes.includes(file.type)) {
            this.showError('Please upload a valid image file (JPG, PNG, or GIF)');
            return;
        }

        // Validate file size (16MB)
        if (file.size > 16 * 1024 * 1024) {
            this.showError('File size exceeds 16MB limit');
            return;
        }

        // Create preview
        const reader = new FileReader();
        reader.onload = (e) => {
            // Show loading state
            this.showLoadingState();
            this.submitForAnalysis(file);
        };
        reader.readAsDataURL(file);
    }

    async submitForAnalysis(file) {
        try {
            // Add realistic delay to simulate processing
            await new Promise(resolve => setTimeout(resolve, 1500));
            
            const formData = new FormData();
            formData.append('image', file);

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Analysis failed');
            }

            const result = await response.json();
            
            // Add another small delay for better UX
            await new Promise(resolve => setTimeout(resolve, 800));
            
            this.displayResults(result);
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError('Analysis failed: ' + error.message);
        }
    }

    showLoadingState() {
        // Hide other states
        document.getElementById('state-idle').hidden = true;
        document.getElementById('state-results').hidden = true;
        
        // Show loading state
        const loadingState = document.getElementById('state-loading');
        loadingState.hidden = false;
        
        // Reset loading steps
        const steps = loadingState.querySelectorAll('.step');
        steps.forEach((step, index) => {
            step.classList.remove('active');
            if (index === 0) {
                step.classList.add('active');
            }
        });
        
        // Animate steps
        this.animateLoadingSteps(steps);
    }

    animateLoadingSteps(steps) {
        let currentStep = 0;
        
        const animate = () => {
            if (currentStep < steps.length) {
                steps.forEach((step, index) => {
                    step.classList.toggle('active', index <= currentStep);
                });
                currentStep++;
                setTimeout(animate, 1200); // Longer delay between steps
            }
        };
        
        setTimeout(animate, 800);
    }

    displayResults(result) {
        this.currentScanId = result.scan_id;
        
        // Update primary result
        document.getElementById('result-label').textContent = result.top_prediction.class;
        document.getElementById('confidence-value').textContent = 
            `${(result.top_prediction.confidence * 100).toFixed(1)}%`;
        
        // Animate confidence bar
        const confidenceBar = document.getElementById('confidence-bar');
        confidenceBar.style.width = '0%';
        setTimeout(() => {
            confidenceBar.style.width = `${result.top_prediction.confidence * 100}%`;
        }, 100);
        
        // Update risk badge
        const riskBadge = document.getElementById('risk-badge');
        riskBadge.className = `risk-badge risk-${result.top_prediction.risk_level}`;
        riskBadge.textContent = result.top_prediction.risk_level === 'high' ? 'High Risk' : 'Low Risk';
        
        // Update uncertainty
        document.getElementById('uncertainty-value').textContent = result.uncertainty.toFixed(3);
        const uncertaintyWarning = document.getElementById('uncertainty-warning');
        uncertaintyWarning.hidden = result.uncertainty < 0.4;
        
        // Update GradCAM
        const gradcamImg = document.getElementById('gradcam-img');
        if (result.gradcam_url) {
            gradcamImg.src = result.gradcam_url;
            gradcamImg.hidden = false;
            document.getElementById('gradcam-overlay').hidden = true;
        } else {
            gradcamImg.hidden = true;
            document.getElementById('gradcam-overlay').hidden = false;
        }
        
        // Update predictions table
        this.updatePredictionsTable(result.predictions);
        
        // Show high-risk alert if needed
        const highRiskBanner = document.getElementById('high-risk-banner');
        highRiskBanner.hidden = !result.high_risk_alert;
        
        // Update download button
        const downloadBtn = document.getElementById('download-report');
        downloadBtn.disabled = !result.report_available;
        downloadBtn.title = result.report_available ? 'Download detailed PDF report' : 'Report generation failed';
        
        // Hide loading and show results
        document.getElementById('state-loading').hidden = true;
        document.getElementById('state-results').hidden = false;
    }

    updatePredictionsTable(predictions) {
        const tableBody = document.getElementById('predictions-body');
        tableBody.innerHTML = '';
        
        predictions.forEach((pred, index) => {
            const row = document.createElement('div');
            row.className = 'prediction-row';
            
            const riskClass = pred.risk_level === 'high' ? 'high' : 'low';
            const riskText = pred.risk_level === 'high' ? 'HIGH' : 'LOW';
            
            row.innerHTML = `
                <div class="rank-cell">#${index + 1}</div>
                <div class="condition-cell">${pred.class}</div>
                <div class="confidence-cell">${(pred.confidence * 100).toFixed(1)}%</div>
                <div class="risk-cell">
                    <span class="risk-indicator ${riskClass}">${riskText}</span>
                </div>
            `;
            
            tableBody.appendChild(row);
        });
    }

    async downloadReport() {
        if (!this.currentScanId) {
            this.showError("No analysis available for download");
            return;
        }
        
        try {
            // Show downloading state
            const downloadBtn = document.getElementById("download-report");
            const originalText = downloadBtn.innerHTML;
            downloadBtn.innerHTML = '<span class="btn-icon">📥</span> Downloading...'
            downloadBtn.disabled = true;
            
            const response = await fetch(`/report/${this.currentScanId}`);
            if (!response.ok) {
                throw new Error(`Report not available (${response.status})`);
            }
            
            // Get filename from Content-Disposition header
            const contentDisposition = response.headers.get("Content-Disposition");
            let filename = `skin_analysis_report_${this.currentScanId}.pdf`;
            if (contentDisposition) {
                const filenameMatch = contentDisposition.match(/filename="?([^"]+)"?$/);
                if (filenameMatch) {
                    filename = filenameMatch[1];
                }
            }
            
            // Create download link
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = filename;
            a.style.display = "none";
            document.body.appendChild(a);
            a.click();
            
            // Clean up
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            // Show success feedback
            downloadBtn.innerHTML = '<span class="btn-icon">✅</span> Download Complete';
            setTimeout(() => {
                downloadBtn.innerHTML = originalText;
                downloadBtn.disabled = false;
            }, 2000);
            
        } catch (error) {
            console.error("Download error:", error);
            this.showError("Failed to download report: " + error.message);
            
            // Reset button state
            const downloadBtn = document.getElementById("download-report");
            downloadBtn.disabled = false;
            downloadBtn.innerHTML = '<span class="btn-icon">📄</span> Download PDF Report';
        }
    }

    resetAnalysis() {
        // Reset file input
        document.getElementById('file-input').value = '';
        
        // Hide all states
        document.getElementById('state-loading').hidden = true;
        document.getElementById('state-results').hidden = true;
        document.getElementById('high-risk-banner').hidden = true;
        
        // Show idle state
        document.getElementById('state-idle').hidden = false;
        
        // Reset current scan ID
        this.currentScanId = null;
    }

    showError(message) {
        // Create error alert
        const errorAlert = document.createElement('div');
        errorAlert.className = 'alert-banner alert-danger';
        errorAlert.innerHTML = `
            <div class="container">
                <div class="alert-content">
                    <span class="alert-icon">❌</span>
                    <div class="alert-text">${message}</div>
                </div>
            </div>
        `;
        
        // Insert after header
        const header = document.querySelector('.header');
        header.parentNode.insertBefore(errorAlert, header.nextSibling);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            errorAlert.remove();
        }, 5000);
        
        // Reset to idle state
        this.resetAnalysis();
    }

    async checkHealth() {
        try {
            const response = await fetch('/api/health');
            const health = await response.json();
            console.log('System health:', health);
        } catch (error) {
            console.warn('Health check failed:', error);
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.skinAIApp = new SkinAIApp();
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl+R or Cmd+R to reset
        if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
            e.preventDefault();
            window.skinAIApp.resetAnalysis();
        }
        
        // Escape to close error messages
        if (e.key === 'Escape') {
            const errorBanners = document.querySelectorAll('.alert-danger');
            errorBanners.forEach(banner => banner.remove());
        }
    });
    
    console.log('.Skin Disease AI App initialized');
    console.log('Ready for skin image analysis');
});

// Global error handler
window.addEventListener('error', (e) => {
    console.error('Global error:', e.error);
});

window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled promise rejection:', e.reason);
});