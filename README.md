# Infant Cry Classification

A deep learning project for classifying infant cries into different categories using audio analysis and convolutional neural networks.

## Dataset

The project uses the DonateACry corpus with 5 classes:
- **belly_pain**: Cries indicating abdominal discomfort
- **burping**: Cries during or related to burping
- **discomfort**: General discomfort cries
- **hungry**: Hunger-related cries
- **tired**: Fatigue-related cries

## Project Structure

```
infant_cry_classification/
├── src/
│   ├── config.py           # Configuration parameters
│   ├── audio_utils.py      # Audio processing utilities
│   ├── dataset.py          # Dataset class and data loading
│   └── model.py            # Neural network architectures
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── requirements.txt        # Python dependencies
├── models/                 # Saved models
├── checkpoints/            # Training checkpoints
├── logs/                   # TensorBoard logs
└── README.md              # This file
```

## Installation

1. Clone the repository and navigate to the project directory:
```bash
cd infant_cry_classification
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

Place your audio dataset in the following structure:
```
infant-cry-audio-corpus/
└── donateacry_corpus/
    ├── belly_pain/
    │   ├── *.wav
    ├── burping/
    │   ├── *.wav
    ├── discomfort/
    │   ├── *.wav
    ├── hungry/
    │   ├── *.wav
    └── tired/
        ├── *.wav
```

Update the `DATASET_PATH` in `src/config.py` if your dataset is located elsewhere.

## Usage

### Training

Train a model using the default CNN architecture:
```bash
python train.py
```

Train with custom parameters:
```bash
python train.py --model cnn --batch-size 32 --epochs 50 --learning-rate 0.001
```

Train using ResNet architecture:
```bash
python train.py --model resnet --epochs 100
```

Available arguments:
- `--data-path`: Path to dataset (default: from config)
- `--model`: Model architecture (`cnn` or `resnet`)
- `--batch-size`: Batch size for training
- `--epochs`: Number of training epochs
- `--learning-rate`: Learning rate for optimizer
- `--save-freq`: Save checkpoint every N epochs

### Evaluation

Evaluate a trained model on the test set:
```bash
python evaluate.py --model-path models/best_cnn.pth --model cnn
```

Evaluate with visualization:
```bash
python evaluate.py --model-path models/best_cnn.pth --model cnn --plot --save-results
```

Available arguments:
- `--data-path`: Path to dataset
- `--model`: Model architecture used
- `--model-path`: Path to trained model checkpoint (required)
- `--batch-size`: Batch size for evaluation
- `--plot`: Generate and save confusion matrix and accuracy plots
- `--save-results`: Save evaluation results to text file

### Monitoring Training

View training progress with TensorBoard:
```bash
tensorboard --logdir logs
```

Then open your browser to `http://localhost:6006`

## Model Architectures

### CNN Model
- 4 convolutional blocks with batch normalization
- Max pooling after each block
- 3 fully connected layers with dropout
- ReLU activation functions

### ResNet Model
- ResNet-inspired architecture with residual connections
- 3 residual layers with increasing channels (64 -> 128 -> 256)
- Adaptive average pooling
- Fully connected output layer

## Audio Processing

The pipeline includes:
1. **Loading**: Audio files are loaded and resampled to 16kHz
2. **Duration normalization**: All clips are padded/trimmed to 3 seconds
3. **Mel spectrogram**: Audio is converted to mel spectrogram (128 mel bands)
4. **Normalization**: Features are normalized using mean and standard deviation
5. **Augmentation** (training only):
   - Random time shifting
   - Random pitch shifting
   - Random noise addition

## Configuration

Key parameters in `src/config.py`:
- `SAMPLE_RATE`: 16000 Hz
- `DURATION`: 3 seconds
- `N_MELS`: 128 mel bands
- `BATCH_SIZE`: 32
- `EPOCHS`: 50
- `LEARNING_RATE`: 0.001
- `VALIDATION_SPLIT`: 0.2
- `TEST_SPLIT`: 0.1

## Results

After training, you will find:
- Best model saved in `models/best_<model_name>.pth`
- Training checkpoints in `checkpoints/`
- TensorBoard logs in `logs/`
- Evaluation results and plots in `models/` (when using `--plot` and `--save-results`)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- librosa
- scikit-learn
- matplotlib
- seaborn
- tensorboard

See `requirements.txt` for complete list.

## License

This project is for research and educational purposes.

## Acknowledgments

Dataset: DonateACry corpus