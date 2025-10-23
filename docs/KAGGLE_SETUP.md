# Kaggle Setup Guide

This guide explains how to run the Wav2Vec2 infant cry classification model in Kaggle.

## Overview

The project now uses the HuggingFace Transformers library with a custom `Wav2Vec2ForSpeechClassification` model, following the proven approach from successful audio classification notebooks.

## Quick Start for Kaggle

The easiest way to use this project in Kaggle is to enable internet access, which allows automatic model downloading.

### Option 1: Enable Internet (Recommended)

**Step 1: Create a Kaggle dataset with the model**

On your local machine with internet:

```bash
# Download the model
python download_model.py --cache-dir ./wav2vec2_model

# This creates a directory with:
# - config.json
# - pytorch_model.bin
# - preprocessor_config.json
# - special_tokens_map.json
# - tokenizer_config.json
# - vocab.json
```

**Step 2: Upload to Kaggle**

1. Zip the `wav2vec2_model` directory
2. Go to Kaggle → Datasets → New Dataset
3. Upload the zip file
4. Name it: `wav2vec2-base-model`

**Step 3: Use in Kaggle Notebook**

```python
# In your Kaggle notebook, update paths in config.py:

# Dataset configuration
DATASET_PATH = "/kaggle/input/infant-cry-audio-corpus/donateacry_corpus"

# Model configuration
WAV2VEC2_MODEL_NAME = "/kaggle/input/wav2vec2-base-model"  # Path to uploaded model
CACHE_DIR = "/kaggle/working/cache"
USE_CACHE = False  # Model already local, no need for cache

# Or use the model directly
WAV2VEC2_MODEL_NAME = "/kaggle/input/wav2vec2-base-model"
```

In Kaggle notebook settings:
1. Click "Settings" (right sidebar)
2. Enable "Internet" toggle
3. Run the training - model will download automatically

The model (`facebook/wav2vec2-base-100k-voxpopuli`) will be automatically downloaded from HuggingFace on first run.

### Option 2: Upload Model as Kaggle Dataset (Offline Mode)

If you need offline mode, you can upload the model as a dataset:

**Step 1: Download model locally (with internet)**
```bash
python download_model.py --cache-dir ./wav2vec2_model
```

**Step 2: Upload to Kaggle**
1. Zip the `wav2vec2_model` directory
2. Go to Kaggle → Datasets → New Dataset
3. Upload and name it `wav2vec2-base-voxpopuli`

**Step 3: Update config in Kaggle**
```python
# In config.py or override in notebook
WAV2VEC2_MODEL_NAME = "/kaggle/input/wav2vec2-base-voxpopuli"
```

## Verification

To check if the model loaded correctly:

```python
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# Test loading
model_path = "/kaggle/input/wav2vec2-base-model"  # or your path
model = Wav2Vec2Model.from_pretrained(model_path, local_files_only=True)
processor = Wav2Vec2Processor.from_pretrained(model_path, local_files_only=True)

print("✓ Model loaded successfully!")
print(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
```

## Troubleshooting

**Error: OSError: Can't load model**
- Solution: Use Option 1 (upload model as Kaggle dataset)

**Error: Out of memory**
- Reduce `BATCH_SIZE` in config.py (try 4 or 2)
- Enable GPU: Settings → Accelerator → GPU T4 x2

**Error: No module named 'transformers'**
```bash
!pip install transformers datasets accelerate
```

**Model download too slow**
- Use Option 1 (pre-download and upload)
- Download on local machine, then upload to Kaggle

## Running in Kaggle Notebook

```python
# Cell 1: Install dependencies (if needed)
!pip install -q transformers datasets accelerate pandas packaging torchaudio librosa

# Cell 2: Clone/upload the project
# Upload your project files or clone from git

# Cell 3: Update config if needed
import sys
sys.path.insert(0, '/kaggle/working/infant_cry_classification/src')

import config
# Update dataset path
config.DATASET_PATH = "/kaggle/input/infant-cry-audio-corpus/donateacry_corpus"

# Reduce batch size if you get OOM errors
config.BATCH_SIZE = 4

# Cell 4: Train
!cd /kaggle/working/infant_cry_classification && python train.py

# Or import and run directly
# from train import train
# trainer = train(dataset_path="/kaggle/input/infant-cry-audio-corpus/donateacry_corpus")
```

## Model Information

Current model: `facebook/wav2vec2-base-100k-voxpopuli`
- Parameters: ~95M
- Memory: ~400MB
- Works well with Kaggle's GPU

Alternative models (change in `config.py`):
- `facebook/wav2vec2-base` - Base model trained on LibriSpeech
- `facebook/wav2vec2-large` - Larger model (requires more memory)
- `facebook/wav2vec2-base-960h` - Fine-tuned on LibriSpeech 960h

## Key Changes from Previous Version

This version uses the proven approach from working audio classification notebooks:

1. **Custom Model Architecture**: Uses `Wav2Vec2ForSpeechClassification` that extends `Wav2Vec2PreTrainedModel`
2. **HuggingFace Datasets**: Uses the `datasets` library for data loading from CSV files
3. **CTCTrainer**: Custom trainer for better training stability
4. **Feature Extractor**: Uses `Wav2Vec2FeatureExtractor` directly instead of `Wav2Vec2Processor`
5. **Simpler Model Loading**: Direct loading with `from_pretrained()` - no complex caching logic

These changes should resolve the previous model downloading issues.
