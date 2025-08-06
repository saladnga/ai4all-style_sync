"""Configuration file for the Outfit Completer application."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "A100"
MODEL_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "cache"

# Dataset paths
AAT_DIR = DATA_DIR / "AAT"
LAT_DIR = DATA_DIR / "LAT"
AAT_IMAGE_DIR = AAT_DIR / "image"
LAT_IMAGE_DIR = LAT_DIR / "image"
AAT_LABEL_PATH = AAT_DIR / "label" / "AAT.json"
LAT_LABEL_PATH = LAT_DIR / "label" / "LAT.json"

# Model configuration
MODEL_CONFIG = {
    "embedding_dim": 128,
    "image_size": (224, 224),
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 3,
    "margin": 1.0,
    "backbone": "resnet18",  # Can be changed to "resnet50", "efficientnet", etc.
}

# Training configuration
TRAIN_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "validation_freq": 10,  # Validate every N batches
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    "page_title": "AI Outfit Completer",
    "page_icon": "👗",
    "layout": "wide",
    "upload_dir": "uploads",
    "max_file_size": 10,  # MB
    "allowed_extensions": ["jpg", "jpeg", "png", "webp"],
}

# Create necessary directories
for directory in [MODEL_DIR, CACHE_DIR, Path(STREAMLIT_CONFIG["upload_dir"])]:
    directory.mkdir(exist_ok=True)
