"""
Configuration file for Facial Expression Detection
"""

import os

# Detect Kaggle Environment
IS_KAGGLE = os.path.exists('/kaggle/input')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
