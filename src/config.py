"""
Configuration file for Facial Expression Detection
"""

import json
import os

# Detect Kaggle Environment
IS_KAGGLE = os.path.exists('/kaggle/input')

# Paths - Project Root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if IS_KAGGLE:
    # Kaggle Paths
    # Note: 'ekspresi-wajah1' must match the dataset name on Kaggle
    # Structure: /kaggle/input/ekspresi-wajah1/dataset/train
    DATASET_DIR = '/kaggle/input/ekspresi-wajah1/dataset'
    TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
    VALIDATION_DIR = os.path.join(DATASET_DIR, 'validation')
    
    # Output directory (writable)
    OUTPUT_DIR = '/kaggle/working'
    MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
else:
    # Local Paths
    DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
    TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
    VALIDATION_DIR = os.path.join(DATASET_DIR, 'validation')
    
    MODELS_DIR = os.path.join(BASE_DIR, 'models')

MODEL_PATH = os.path.join(MODELS_DIR, 'expression_model.h5')
LABELS_PATH = os.path.join(MODELS_DIR, 'class_labels.json')

# Model Settings
INPUT_SHAPE = (48, 48, 1)
NUM_CLASSES = 7
USE_TRANSFER_LEARNING = True  # Set to True to use MobileNetV2

# Default Labels
DEFAULT_LABELS = {
    '0': 'marah',
    '1': 'jijik',
    '2': 'takut',
    '3': 'senang',
    '4': 'netral',
    '5': 'sedih',
    '6': 'kaget'
}

# Colors (BGR Format)
COLORS = {
    'marah': (0, 0, 255),      # Red
    'jijik': (0, 128, 128),    # Teal
    'takut': (128, 0, 128),    # Purple
    'senang': (0, 255, 0),     # Green
    'netral': (255, 255, 255), # White
    'sedih': (255, 0, 0),      # Blue
    'kaget': (0, 255, 255)     # Yellow
}

# Camera Settings
CAMERA_INDEX = 0
WINDOW_NAME = 'Facial Expression Detection'

# Face Detection Settings
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5


def load_labels():
    """Mapping {str(idx): label} dari LABELS_PATH, fallback DEFAULT_LABELS.

    Sumber kebenaran adalah file yang ditulis train.py dari
    flow_from_directory.class_indices (urutan alfabetis) — jangan hardcode
    DEFAULT_LABELS di consumer. Dukung dua format: {idx: label} (train.py)
    dan legacy {label: idx}. (json + os = stdlib ringan, aman di top-level.)
    """
    fallback = dict(DEFAULT_LABELS)
    if not LABELS_PATH or not os.path.exists(LABELS_PATH):
        return fallback
    try:
        with open(LABELS_PATH, "r") as f:
            data = json.load(f)
        if data and all(str(k).isdigit() for k in data.keys()):
            normalized = {str(k): str(v) for k, v in data.items()}
        else:
            normalized = {str(v): str(k) for k, v in data.items()}
        if len(normalized) != int(NUM_CLASSES):
            return fallback
        return normalized
    except (OSError, ValueError):
        return fallback

