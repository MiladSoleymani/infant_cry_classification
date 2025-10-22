"""
Configuration file for infant cry classification project
"""
import os

# Dataset configuration
DATASET_PATH = "infant-cry-audio-corpus/donateacry_corpus"
CLASSES = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
NUM_CLASSES = len(CLASSES)

# Audio processing configuration
SAMPLE_RATE = 16000  # Target sample rate
DURATION = 3  # Fixed duration in seconds
N_MELS = 128  # Number of mel bands
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Hop length for STFT

# Model configuration
INPUT_SHAPE = (N_MELS, int(SAMPLE_RATE * DURATION / HOP_LENGTH) + 1)

# Training configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# Paths
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
MODEL_DIR = "models"

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
