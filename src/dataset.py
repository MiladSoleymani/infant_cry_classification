"""
Dataset preparation for Wav2Vec2-based infant cry classification
Using HuggingFace datasets library (based on working notebook implementation)
"""
import os
from pathlib import Path
from typing import Dict, List, Union, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
from transformers import Wav2Vec2FeatureExtractor
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split

import config


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Based on DataCollatorCTCWithPadding from the working notebook
    """
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for batching

        Args:
            features: List of feature dictionaries with input_values and labels

        Returns:
            Batched dictionary with input_values and labels tensors
        """
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features, dtype=d_type)

        return batch


def prepare_dataset_csv(
    dataset_path: str = config.DATASET_PATH,
    save_path: str = ".",
    val_split: float = config.VALIDATION_SPLIT,
    test_split: float = config.TEST_SPLIT,
    random_seed: int = config.RANDOM_SEED
):
    """
    Prepare CSV files for dataset loading
    Similar to the notebook's approach

    Args:
        dataset_path: Path to dataset root directory
        save_path: Path to save CSV files
        val_split: Validation set proportion
        test_split: Test set proportion
        random_seed: Random seed for reproducibility

    Returns:
        Paths to train, validation, and test CSV files
    """
    data = []

    # Create label mapping
    class_to_idx = {class_name: idx for idx, class_name in enumerate(config.CLASSES)}

    # Iterate through each class directory and collect file paths
    for class_name in config.CLASSES:
        class_dir = os.path.join(dataset_path, class_name)

        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist")
            continue

        # Get all .wav files in the class directory
        wav_files = list(Path(class_dir).glob('*.wav'))

        for wav_file in wav_files:
            data.append({
                "path": str(wav_file),
                "label": class_name
            })

    # Create DataFrame
    df = pd.DataFrame(data)
    print(f"\nLoaded {len(df)} audio files from {len(config.CLASSES)} classes")
    print(df.groupby("label").count()[["path"]])

    # Split data into train, validation, and test sets
    train_df, rest_df = train_test_split(
        df,
        test_size=(val_split + test_split),
        random_state=random_seed,
        stratify=df["label"]
    )

    val_size_adjusted = val_split / (val_split + test_split)
    val_df, test_df = train_test_split(
        rest_df,
        test_size=(1 - val_size_adjusted),
        random_state=random_seed,
        stratify=rest_df["label"]
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save to CSV
    os.makedirs(save_path, exist_ok=True)
    train_csv = os.path.join(save_path, "train.csv")
    val_csv = os.path.join(save_path, "validation.csv")
    test_csv = os.path.join(save_path, "test.csv")

    train_df.to_csv(train_csv, encoding="utf-8", index=False)
    val_df.to_csv(val_csv, encoding="utf-8", index=False)
    test_df.to_csv(test_csv, encoding="utf-8", index=False)

    print(f"\nDataset splits:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    return train_csv, val_csv, test_csv


def speech_file_to_array_fn(path: str, target_sampling_rate: int) -> np.ndarray:
    """
    Load and resample audio file to target sampling rate

    Args:
        path: Path to audio file
        target_sampling_rate: Target sampling rate (16000 for Wav2Vec2)

    Returns:
        Audio array
    """
    try:
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech
    except Exception as e:
        print(f"Error loading {path}: {e}")
        # Return silence if loading fails
        return np.zeros(target_sampling_rate)


def label_to_id(label: str, label_list: List[str]) -> int:
    """
    Convert label string to ID

    Args:
        label: Label string
        label_list: List of all labels

    Returns:
        Label ID
    """
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1
    return label


def preprocess_function(
    examples: Dict,
    input_column: str,
    output_column: str,
    label_list: List[str],
    feature_extractor: Wav2Vec2FeatureExtractor
) -> Dict:
    """
    Preprocess function for dataset mapping
    Based on the notebook's approach

    Args:
        examples: Batch of examples from dataset
        input_column: Name of input column (path)
        output_column: Name of output column (label)
        label_list: List of all labels
        feature_extractor: Wav2Vec2FeatureExtractor instance

    Returns:
        Processed batch with input_values and labels
    """
    target_sampling_rate = feature_extractor.sampling_rate

    speech_list = [
        speech_file_to_array_fn(path, target_sampling_rate)
        for path in examples[input_column]
    ]
    target_list = [
        label_to_id(label, label_list)
        for label in examples[output_column]
    ]

    result = feature_extractor(speech_list, sampling_rate=target_sampling_rate)
    result["labels"] = list(target_list)

    return result


def load_and_prepare_datasets(
    dataset_path: str = config.DATASET_PATH,
    save_path: str = ".",
    model_name: str = config.WAV2VEC2_MODEL_NAME
) -> tuple:
    """
    Load and prepare datasets using HuggingFace datasets library
    Following the notebook's approach

    Args:
        dataset_path: Path to dataset root directory
        save_path: Path to save CSV files
        model_name: Wav2Vec2 model name for feature extractor

    Returns:
        Tuple of (train_dataset, eval_dataset, feature_extractor, label_list, num_labels)
    """
    # Prepare CSV files
    train_csv, val_csv, test_csv = prepare_dataset_csv(dataset_path, save_path)

    # Load datasets from CSV
    data_files = {
        "train": train_csv,
        "validation": val_csv,
        "test": test_csv,
    }

    dataset = load_dataset("csv", data_files=data_files)

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    # Define input and output columns
    input_column = "path"
    output_column = "label"

    # Get unique labels
    label_list = train_dataset.unique(output_column)
    label_list.sort()  # Sort for determinism
    num_labels = len(label_list)

    print(f"\nA classification problem with {num_labels} classes:")
    print(f"  {label_list}")

    # Load feature extractor
    print(f"\nLoading feature extractor from: {model_name}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    target_sampling_rate = feature_extractor.sampling_rate
    print(f"Target sampling rate: {target_sampling_rate}")

    # Create preprocessing function with fixed parameters
    def preprocess_fn(examples):
        return preprocess_function(
            examples,
            input_column,
            output_column,
            label_list,
            feature_extractor
        )

    # Map preprocessing to datasets
    print("\nPreprocessing training dataset...")
    train_dataset = train_dataset.map(
        preprocess_fn,
        batch_size=10,
        batched=True,
        remove_columns=[input_column]
    )

    print("Preprocessing validation dataset...")
    eval_dataset = eval_dataset.map(
        preprocess_fn,
        batch_size=10,
        batched=True,
        remove_columns=[input_column]
    )

    print("Preprocessing test dataset...")
    test_dataset = test_dataset.map(
        preprocess_fn,
        batch_size=10,
        batched=True,
        remove_columns=[input_column]
    )

    return train_dataset, eval_dataset, test_dataset, feature_extractor, label_list, num_labels


def compute_metrics(p):
    """
    Compute metrics for evaluation

    Args:
        p: EvalPrediction object

    Returns:
        Dictionary with accuracy metric
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)

    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
