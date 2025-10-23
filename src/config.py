"""
Configuration file for infant cry classification project using Wav2Vec2
"""
import os

# Dataset configuration
DATASET_PATH = "/kaggle/input/infant-cry-audio-corpus/donateacry_corpus"
CLASSES = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
NUM_CLASSES = len(CLASSES)

# Wav2Vec2 model configuration
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base"  # Pre-trained model
# Alternative models:
# "facebook/wav2vec2-large" - Larger model, better performance
# "facebook/wav2vec2-base-960h" - Fine-tuned on LibriSpeech

# Audio processing configuration
SAMPLE_RATE = 16000  # Wav2Vec2 requires 16kHz
MAX_DURATION = 5.0  # Maximum duration in seconds
TARGET_LENGTH = int(SAMPLE_RATE * MAX_DURATION)  # 80000 samples

# Training configuration
BATCH_SIZE = 8  # Smaller batch size for Wav2Vec2 (memory intensive)
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 8 * 4 = 32
EPOCHS = 20  # Fewer epochs needed with pre-trained model
LEARNING_RATE = 3e-5  # Lower learning rate for fine-tuning
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# Model training settings
FREEZE_FEATURE_EXTRACTOR = True  # Freeze Wav2Vec2 CNN layers initially
FREEZE_EPOCHS = 5  # Unfreeze after this many epochs
DROPOUT = 0.1
ATTENTION_DROPOUT = 0.1
HIDDEN_DROPOUT = 0.1

# Class balancing configuration
USE_CLASS_WEIGHTS = True  # Use weighted loss function
USE_WEIGHTED_SAMPLER = True  # Use weighted random sampler for training

# Paths
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
MODEL_DIR = "models"

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
