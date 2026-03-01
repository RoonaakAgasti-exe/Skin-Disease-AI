# Skin Disease AI

AI-powered skin disease classification using an ensemble of three deep learning models with GradCAM visualisation, uncertainty quantification, and PDF reporting.

<img width="1428" height="910" alt="Screenshot 2026-03-01 195052" src="https://github.com/user-attachments/assets/1c766460-1953-48c4-9909-ae2e16de0932" />
<img width="780" height="894" alt="Screenshot 2026-03-01 195111" src="https://github.com/user-attachments/assets/373af9cf-659e-42a0-9dbe-57253d014f24" />
<img width="351" height="712" alt="Screenshot 2026-03-01 195257" src="https://github.com/user-attachments/assets/29375ab1-46bd-460b-8ab9-e245eea90f8d" />

> ⚠️ **Medical Disclaimer** — For educational and decision support purposes only. Not a substitute for professional medical diagnosis. Not FDA-approved for clinical use.

---

## Overview

Classifies skin conditions across **22 classes** using an ensemble of EfficientNetV2-M, ConvNeXt-Tiny, and Vision Transformer (ViT) models. Includes a Flask web app with drag-and-drop upload, GradCAM attention heatmaps, and downloadable PDF reports.

---

## Classes

| # | Condition | # | Condition |
|---|-----------|---|-----------|
| 1 | Acne | 12 | Psoriasis |
| 2 | Actinic Keratosis ⚠ | 13 | Rosacea |
| 3 | Benign Tumors | 14 | Seborrhoeic Keratoses |
| 4 | Bullous | 15 | Skin Cancer ⚠ |
| 5 | Candidiasis | 16 | Sun / Sunlight Damage |
| 6 | Drug Eruption | 17 | Tinea |
| 7 | Eczema | 18 | Unknown / Normal |
| 8 | Infestations & Bites | 19 | Vascular Tumors |
| 9 | Lichen | 20 | Vasculitis |
| 10 | Lupus | 21 | Vitiligo |
| 11 | Moles | 22 | Warts |

⚠ = automatically flagged as high-risk with urgent-care warning.

---

## Features

- **Ensemble model** — EfficientNetV2-M (40%) + ConvNeXt-Tiny (35%) + ViT (25%)
- **GradCAM++ heatmaps** — highlights which skin region influenced the prediction
- **Uncertainty quantification** — entropy-based confidence measurement
- **High-risk alerts** — automatic warnings for malignant conditions
- **PDF reports** — downloadable report with images, scores, and recommendations
- **Dark clinical UI** — drag-and-drop upload, real-time results, mobile-friendly

---

## Performance

| Metric | Score |
|--------|-------|
| Overall accuracy | ~85–90% |
| High-risk sensitivity | > 95% |

---

## Installation

**Requirements:** Python 3.8+, 8GB RAM, GPU optional

```bash
git clone https://github.com/yourusername/skin-disease-ai.git
cd skin-disease-ai
pip install -r requirements.txt
```

**Dataset structure:**

Download "Skin-Diseases" dataset from kaggle

## Usage

**Train:**
```bash
python train.py                        # all models
python train.py --model efficientnetv2 # single model
```

**Run app:**
```bash
python app.py
# → http://localhost:5000
```

**API:**
```python
import requests

with open('image.jpg', 'rb') as f:
    res = requests.post('http://localhost:5000/predict', files={'image': f})

print(res.json()['top_prediction'])
# {'class': 'Eczema', 'confidence': 0.847, 'risk_level': 'low'}
```

**Endpoints:**
```
POST /predict          → run inference on uploaded image
GET  /report/<scan_id> → download PDF report
GET  /api/health       → system health check
GET  /api/classes      → list all 22 classes
```

---

## Project Structure

```
skin-disease-ai/
├── app.py               # Flask server + API endpoints
├── models.py            # EfficientNetV2, ConvNeXt, ViT architectures
├── predictor.py         # Ensemble inference logic
├── data_loader.py       # Preprocessing + augmentation pipeline
├── gradcam.py           # GradCAM++ visualisation
├── report_generator.py  # PDF report generation (fpdf2)
├── train.py             # Full training pipeline
├── simple_train.py      # Simplified sklearn-based training
├── config.py            # Model weights, class names, settings
├── templates/index.html # Web UI
├── static/css/          # Stylesheets
├── static/js/           # Frontend JavaScript
└── requirements.txt
```

---

## Configuration

Edit `config.py` to adjust:

```python
ENSEMBLE_WEIGHTS = {'efficientnetv2': 0.40, 'convnext': 0.35, 'vit': 0.25}
HIGH_RISK_CLASSES = ['SkinCancer', 'Actinic_Keratosis']
MAX_UPLOAD_SIZE   = 10 * 1024 * 1024   # 10MB
```

---

## Tech Stack

**Backend:** Flask · TensorFlow 2.15 · Keras · OpenCV · fpdf2  
**Frontend:** Vanilla JS · CSS variables · no framework dependencies  
**Training:** Focal Loss · class-balanced weights · two-phase fine-tuning

---

*Built for advancing dermatological AI research.*
