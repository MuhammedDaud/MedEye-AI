"""
╔════════════════════════════════════════════════════════════════════════════╗
║                            MEDEYE v2.0                                      ║
║                   Medical Eye Disease Detection System                       ║
║                                                                              ║
║  Developer: Muhammad Daud                                                    ║
║  Project Type: Full Intellectual Property (FIP)                             ║
║  Created: 2026                                                               ║
║  License: Proprietary                                                        ║
║                                                                              ║
║  Configuration Module - Centralized settings and parameters                 ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS (Portable)
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for directory in [MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# Class information (computed dynamically, but set as defaults)
CLASS_NAMES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
NUM_CLASSES = len(CLASS_NAMES)

# Expected class distribution (for validation)
EXPECTED_CLASS_DISTRIBUTION = {
    "cataract": 1038,
    "diabetic_retinopathy": 1098,
    "glaucoma": 1007,
    "normal": 1074,
}

TOTAL_IMAGES = sum(EXPECTED_CLASS_DISTRIBUTION.values())  # 4217

# ============================================================================
# IMAGE PROCESSING
# ============================================================================
# For Transfer Learning Models (EfficientNet, MobileNet, ResNet, etc.)
TRANSFER_LEARNING_IMG_SIZE = (224, 224)
TRANSFER_LEARNING_IMG_CHANNELS = 3  # RGB

# For Traditional ML Models (SVM, Random Forest)
TRADITIONAL_ML_IMG_SIZE = (128, 128)
TRADITIONAL_ML_IMG_CHANNELS = 1  # Grayscale

# Image preprocessing
IMAGE_NORMALIZATION = 1.0 / 255.0  # Normalize to [0, 1]
VALID_IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "tiff", "tif"]

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
# Default training parameters
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_OPTIMIZER = "adam"
DEFAULT_LOSS = "categorical_crossentropy"
DEFAULT_METRICS = ["accuracy"]

# Data split
TRAIN_TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.1
RANDOM_STATE = 42

# Data augmentation
DATA_AUGMENTATION_CONFIG = {
    "rotation_range": 20,
    "width_shift_range": 0.15,
    "height_shift_range": 0.15,
    "shear_range": 0.2,
    "zoom_range": 0.2,
    "horizontal_flip": False,
    "vertical_flip": True,
    "fill_mode": "nearest",
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Transfer Learning Models
TRANSFER_LEARNING_MODELS = [
    "efficientnetb3",
    "mobilenet",
    "densenet121",
    "resnet50",
    "vgg16",
    "xception",
    "inceptionv3",
]

# Traditional ML Models
TRADITIONAL_ML_MODELS = [
    "svm",
    "random_forest",
    "baseline_cnn",
]

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = LOGS_DIR / "medeye.log"

# ============================================================================
# ERROR HANDLING
# ============================================================================
MAX_RETRY_ATTEMPTS = 3
SKIP_CORRUPTED_IMAGES = True  # Skip instead of failing
LOG_CORRUPTED_IMAGES = True   # Log corrupted image paths

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
CONFIDENCE_THRESHOLD = 0.0  # Accept all predictions (0.0-1.0)
CONFIDENCE_DECIMAL_PLACES = 2

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
GPU_MEMORY_FRACTION = 0.8  # Use up to 80% of GPU memory
ENABLE_MIXED_PRECISION = True  # For faster training on modern GPUs

# ============================================================================
# VALIDATION THRESHOLDS
# ============================================================================
MIN_IMAGE_WIDTH = 50
MIN_IMAGE_HEIGHT = 50
MAX_IMAGE_SIZE_MB = 50  # Maximum image file size in MB

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_class_names():
    """Get class names (can be extended to read from config file)."""
    return CLASS_NAMES


def get_class_index(class_name):
    """Get numeric index of a class."""
    if class_name not in CLASS_NAMES:
        raise ValueError(f"Unknown class: {class_name}. Valid: {CLASS_NAMES}")
    return CLASS_NAMES.index(class_name)


def get_data_dir():
    """Get dataset directory path."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")
    return DATA_DIR


def verify_config(skip_data_dir_check: bool = False):
    """
    Verify configuration is valid.
    
    Args:
        skip_data_dir_check (bool): Skip checking if dataset directory exists
    
    Returns:
        bool: True if configuration is valid
    
    Raises:
        ValueError: If configuration is invalid
    """
    errors = []
    
    if not skip_data_dir_check and not DATA_DIR.exists():
        errors.append(f"Dataset directory not found: {DATA_DIR}")
    
    if TRAIN_TEST_SPLIT <= 0 or TRAIN_TEST_SPLIT >= 1:
        errors.append("TRAIN_TEST_SPLIT must be between 0 and 1")
    
    if DEFAULT_BATCH_SIZE <= 0:
        errors.append("DEFAULT_BATCH_SIZE must be positive")
    
    if DEFAULT_EPOCHS <= 0:
        errors.append("DEFAULT_EPOCHS must be positive")
    
    if DEFAULT_LEARNING_RATE <= 0:
        errors.append("DEFAULT_LEARNING_RATE must be positive")
    
    if len(CLASS_NAMES) != NUM_CLASSES:
        errors.append("CLASS_NAMES length doesn't match NUM_CLASSES")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))
    
    return True


# Verify configuration parameters on import (but not data directory to allow module imports)
try:
    verify_config(skip_data_dir_check=True)
except ValueError as e:
    import warnings
    warnings.warn(f"Configuration validation warning: {e}")
