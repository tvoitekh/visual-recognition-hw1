"""
Shared configuration for plant classification models
"""
import os
import torch

# Data paths
DATA_ROOT = './data/'
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VAL_DIR = os.path.join(DATA_ROOT, 'val')
TEST_DIR = os.path.join(DATA_ROOT, 'test')

# Training parameters
BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True
NUM_EPOCHS = 60
PATIENCE = 30
LABEL_SMOOTHING = 0.1
INITIAL_LR = 0.0001
MIN_LR = 1e-7
WEIGHT_DECAY = 0.01
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
USE_MIXUP = True
USE_CUTMIX = True
GRAD_CLIP_VALUE = 1.0

# Model parameters
NUM_CLASSES = len(os.listdir(TRAIN_DIR)) if os.path.exists(TRAIN_DIR) else 100
DROPOUT_RATE = 0.3

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging
SAVE_DIR = './models/'
os.makedirs(SAVE_DIR, exist_ok=True)

# TTA configuration
NUM_TTA = 10

# Ensemble configuration
ENSEMBLE_WEIGHTS = {
    'resnet34': 0.2,
    'resnet50': 0.3,
    'resnext50': 0.5,
}
