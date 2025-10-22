"""
Dataset class for infant cry classification
"""
import os
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import config
from audio_utils import preprocess_audio, load_audio, audio_to_melspectrogram, augment_audio


class InfantCryDataset(Dataset):
    """
    PyTorch Dataset for infant cry classification
    """

    def __init__(self, file_paths: List[str], labels: List[int],
                 augment: bool = False):
        """
        Initialize dataset

        Args:
            file_paths: List of audio file paths
            labels: List of corresponding labels
            augment: Whether to apply data augmentation
        """
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample

        Args:
            idx: Sample index

        Returns:
            Tuple of (mel_spectrogram, label)
        """
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load audio
        audio = load_audio(file_path)

        # Apply augmentation if training
        if self.augment:
            audio = augment_audio(audio)

        # Convert to mel spectrogram
        mel_spec = audio_to_melspectrogram(audio)

        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

        # Convert to tensor and add channel dimension
        mel_spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)

        return mel_spec_tensor, label


def load_dataset(dataset_path: str = config.DATASET_PATH) -> Tuple[List[str], List[int]]:
    """
    Load all audio files and labels from dataset directory

    Args:
        dataset_path: Path to dataset root directory

    Returns:
        Tuple of (file_paths, labels)
    """
    file_paths = []
    labels = []

    # Create label mapping
    class_to_idx = {class_name: idx for idx, class_name in enumerate(config.CLASSES)}

    # Iterate through each class directory
    for class_name in config.CLASSES:
        class_dir = os.path.join(dataset_path, class_name)

        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist")
            continue

        # Get all .wav files in the class directory
        wav_files = list(Path(class_dir).glob('*.wav'))

        for wav_file in wav_files:
            file_paths.append(str(wav_file))
            labels.append(class_to_idx[class_name])

    print(f"Loaded {len(file_paths)} audio files from {len(config.CLASSES)} classes")
    for class_name, idx in class_to_idx.items():
        count = labels.count(idx)
        print(f"  {class_name}: {count} samples")

    return file_paths, labels


def create_data_splits(file_paths: List[str], labels: List[int],
                      val_split: float = config.VALIDATION_SPLIT,
                      test_split: float = config.TEST_SPLIT,
                      random_seed: int = config.RANDOM_SEED) -> Tuple:
    """
    Split data into train, validation, and test sets

    Args:
        file_paths: List of audio file paths
        labels: List of corresponding labels
        val_split: Validation set proportion
        test_split: Test set proportion
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_files, train_labels, val_files, val_labels, test_files, test_labels)
    """
    # First split: separate test set
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        file_paths, labels, test_size=test_split, random_state=random_seed, stratify=labels
    )

    # Second split: separate validation from training
    val_size_adjusted = val_split / (1 - test_split)
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels, test_size=val_size_adjusted,
        random_state=random_seed, stratify=train_val_labels
    )

    print(f"\nDataset splits:")
    print(f"  Training: {len(train_files)} samples")
    print(f"  Validation: {len(val_files)} samples")
    print(f"  Test: {len(test_files)} samples")

    return train_files, train_labels, val_files, val_labels, test_files, test_labels


def get_data_loaders(batch_size: int = config.BATCH_SIZE,
                    dataset_path: str = config.DATASET_PATH):
    """
    Create PyTorch DataLoaders for train, validation, and test sets

    Args:
        batch_size: Batch size for data loaders
        dataset_path: Path to dataset root directory

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load dataset
    file_paths, labels = load_dataset(dataset_path)

    # Create splits
    train_files, train_labels, val_files, val_labels, test_files, test_labels = \
        create_data_splits(file_paths, labels)

    # Create datasets
    train_dataset = InfantCryDataset(train_files, train_labels, augment=True)
    val_dataset = InfantCryDataset(val_files, val_labels, augment=False)
    test_dataset = InfantCryDataset(test_files, test_labels, augment=False)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader, test_loader
