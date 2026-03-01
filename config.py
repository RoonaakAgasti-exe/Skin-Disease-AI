import os

# Dataset Classes - Updated to match archive folder structure
CLASSES = [
    "Acne", "Actinic_Keratosis", "Benign_tumors", "Bullous", "Candidiasis",
    "DrugEruption", "Eczema", "Infestations_Bites", "Lichen", "Lupus",
    "Moles", "Psoriasis", "Rosacea", "Seborrh_Keratoses", "SkinCancer",
    "Sun_Sunlight_Damage", "Tinea", "Unknown_Normal", "Vascular_Tumors",
    "Vasculitis", "Vitiligo", "Warts"
]

NUM_CLASSES = len(CLASSES)

# High-risk conditions that require immediate medical attention
HIGH_RISK = {"SkinCancer", "Actinic_Keratosis"}

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive", "SkinDisease", "SkinDisease")
TRAIN_DIR = os.path.join(ARCHIVE_DIR, "train")
TEST_DIR = os.path.join(ARCHIVE_DIR, "test")

# Output directories
MODELS_DIR = os.path.join(BASE_DIR, "models_saved")
os.makedirs(MODELS_DIR, exist_ok=True)

UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Image preprocessing parameters
IMG_SIZE_EFFNET = (384, 384)
IMG_SIZE_CONVNEXT = (384, 384)
IMG_SIZE_VIT = (224, 224)
IMG_SIZE_CUSTOM = (224, 224)

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50
VALIDATION_SPLIT = 0.2

# Model weights for ensemble
ENSEMBLE_WEIGHTS = {
    "efficientnet": 0.40,
    "convnext": 0.35,
    "vit": 0.25
}

# Advanced training parameters
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 0.0001
DROPOUT_RATE = 0.3