# 🩺 SkinAI Clinical Gateway: Advanced Medical Diagnosis System

<img width="1090" height="730" alt="Screenshot 2026-03-07 175511" src="https://github.com/user-attachments/assets/3ffc097b-dc5e-4ff1-bc36-5e9f090a0d1b" />


**SkinAI Clinical Gateway** is a professional-grade dermatological diagnostic support tool. It leverages **Zero-Shot Vision Transformers (ViT)** and **Advanced Computer Vision** to provide clinical-grade analysis of skin conditions instantly.

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Technical Architecture](#-technical-architecture)
- [Clinical Knowledge Base](#-clinical-knowledge-base)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Safety & Disclaimer](#-safety--disclaimer)

---

## 🌟 Project Overview
The gateway bridges the gap between complex AI and clinical practice. By using natural language supervision (CLIP), the system identifies skin pathologies without requiring local datasets, making it a flexible and powerful tool for medical research and preliminary screening.

---

## 🚀 Key Features
- **Zero-Shot AI Core**: Powered by `openai/clip-vit-large-patch14`. No local training required.
- **DullRazor Preprocessing**: Automatic hair removal for clearer lesion analysis.
- **Gray World Restoration**: Illumination normalization for unbiased diagnostics.
- **Live Diagnostics**: Real-time probability visualization via **Chart.js**.
- **Automated Reporting**: Instant PDF generation with clinical metadata and preprocessed visual evidence.
- **Medical Dashboard**: Professional, clean UI designed for healthcare environments.

---

## 🏗 Technical Architecture

### 🧠 The AI Inference Engine
The system utilizes **CLIP (Contrastive Language-Image Pre-training)** to perform zero-shot classification:
- **Zero-Shot Capability**: Recognizes skin conditions by correlating visual patterns with clinical text descriptions.
- **Ensemble-Lite Logic**: Optimized single-prompt inference for rapid response times.

### 🖼 Medical Computer Vision
Two critical classical CV algorithms ensure high-fidelity input:
1. **DullRazor**: Morphological Blackhat filtering identifies hair artifacts, which are then inpainted using telea algorithms.
2. **Gray World**: Estimates and corrects color cast based on the "Gray World" hypothesis, ensuring standard clinical color representation.

---

## 🔬 Clinical Knowledge Base
The diagnostic engine covers 22 clinical classes including:
- **Melanoma & Skin Cancer** (High Risk)
- **Actinic Keratosis** (Pre-cancerous)
- **Chronic Inflammatory**: Eczema, Psoriasis, Rosacea.
- **Infections**: Candidiasis, Tinea, Warts.

Each result is paired with:
- ✅ **Standard Medicines**: Common clinical treatments (e.g., Metronidazole, Retinoids).
- ✅ **Preventive Strategy**: Evidence-based lifestyle adjustments.

---

## 💻 Installation & Setup

### 1. Requirements
- Python 3.9+
- 8GB+ RAM
- Windows/Linux/macOS

### 2. Quick Install
```bash
# Clone the repository
git clone https://github.com/your-repo/skin-disease-ai.git
cd skin-disease-ai

# Install dependencies
pip install -r requirements.txt
```

### 3. Run
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📄 Project Structure
```text
├── app.py                # Main Flask Application
├── predictor.py          # AI & CV Preprocessing Logic
├── report_generator.py   # PDF Generation Service
├── config.py             # Medical DB & Configuration
├── static/
│   ├── css/style.css     # Medical UI Design
│   └── js/app.js         # Frontend Logic & Charts
└── templates/
    └── index.html        # Clinical Dashboard
```

---

## ⚠️ Safety & Disclaimer
> [!WARNING]
> This software is for **Educational and Clinical Support Purposes Only**.
> 1. AI models can produce **False Negatives** or **False Positives**.
> 2. This is **NOT** a definitive medical diagnosis.
> 3. Results **MUST** be verified by a board-certified dermatologist.
> 4. In case of emergency or rapidly changing skin lesions, seek professional care immediately.

---

## 📜 License
Distributed under the **MIT License**. See `LICENSE` for more information.
