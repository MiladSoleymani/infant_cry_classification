# Kaggle/Offline Setup Guide

This guide explains how to use the Wav2Vec2 model in offline environments like Kaggle.

## Problem

The error `Can't load the model for 'facebook/wav2vec2-base'` occurs when:
- No internet connection (Kaggle notebooks in offline mode)
- Firewall blocking HuggingFace
- Model not cached locally

## Solution

### Option 1: Download Model First (Recommended for Kaggle)

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

### Option 2: Enable Internet in Kaggle

In Kaggle notebook settings:
1. Click "Settings" (right sidebar)
2. Enable "Internet" toggle
3. Run the training - model will download automatically

**Note**: This uses Kaggle's internet quota.

### Option 3: Use Pre-cached Models

If you've run the notebook before:

```python
# config.py
CACHE_DIR = "/kaggle/working/wav2vec2_cache"
USE_CACHE = True
```

The model will be saved in the cache directory after first download.

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

## Quick Start for Kaggle

```python
# Cell 1: Install dependencies
!pip install -q transformers datasets accelerate

# Cell 2: Update config
import sys
sys.path.insert(0, '/kaggle/working/infant_cry_classification/src')

# Modify config.py or override:
import config
config.DATASET_PATH = "/kaggle/input/infant-cry-audio-corpus/donateacry_corpus"
config.WAV2VEC2_MODEL_NAME = "/kaggle/input/wav2vec2-base-model"  # If uploaded
config.BATCH_SIZE = 4  # Reduce if OOM

# Cell 3: Train
!cd /kaggle/working/infant_cry_classification && python train.py
```

## Model Sizes

| Model | Parameters | Memory | Recommended |
|-------|-----------|---------|-------------|
| wav2vec2-base | 95M | ~400MB | ✓ Good for Kaggle |
| wav2vec2-large | 317M | ~1.2GB | Requires GPU |
| wav2vec2-base-960h | 95M | ~400MB | Fine-tuned version |
