"""
Dataset class for Wav2Vec2-based infant cry classification
"""
import os
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import config
from audio_utils import load_audio, pad_or_truncate, augment_audio


class InfantCryDataset(Dataset):
    """
    PyTorch Dataset for infant cry classification using raw audio
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
            Tuple of (audio_tensor, label)
        """
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load audio
        audio = load_audio(file_path)

        # Apply augmentation if training
        if self.augment:
            audio = augment_audio(audio)

        # Pad or truncate to fixed length
        audio = pad_or_truncate(audio)

        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio)

        return audio_tensor, label


def collate_fn_wav2vec2(batch):
    """
    Custom collate function for Wav2Vec2 that handles batching properly

    Args:
        batch: List of (audio, label) tuples

    Returns:
        Dictionary with input_values, attention_mask, and labels
    """
    from audio_utils import get_processor

    audios = [item[0].numpy() for item in batch]
    labels = [item[1] for item in batch]

    # Process batch with Wav2Vec2 processor
    processor = get_processor()
    inputs = processor(
        audios,
        sampling_rate=config.SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )

    inputs['labels'] = torch.tensor(labels, dtype=torch.long)

    return inputs


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


def compute_class_weights(labels: List[int]) -> torch.Tensor:
    """
    Compute class weights for imbalanced dataset

    Args:
        labels: List of labels

    Returns:
        Tensor of class weights
    """
    labels_array = np.array(labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels_array),
        y=labels_array
    )

    print(f"\nClass weights for loss function:")
    for i, class_name in enumerate(config.CLASSES):
        print(f"  {class_name}: {class_weights[i]:.4f}")

    return torch.FloatTensor(class_weights)


def create_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """
    Create weighted random sampler for imbalanced dataset

    Args:
        labels: List of training labels

    Returns:
        WeightedRandomSampler instance
    """
    # Count samples per class
    class_counts = np.bincount(labels)

    # Calculate weights (inverse of class frequency)
    class_weights = 1.0 / class_counts

    # Assign weight to each sample based on its class
    sample_weights = [class_weights[label] for label in labels]

    print(f"\nWeighted sampler statistics:")
    for i, class_name in enumerate(config.CLASSES):
        print(f"  {class_name}: count={class_counts[i]}, weight={class_weights[i]:.4f}")

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


def get_data_loaders(batch_size: int = config.BATCH_SIZE,
                    dataset_path: str = config.DATASET_PATH,
                    use_weighted_sampler: bool = config.USE_WEIGHTED_SAMPLER):
    """
    Create PyTorch DataLoaders for train, validation, and test sets

    Args:
        batch_size: Batch size for data loaders
        dataset_path: Path to dataset root directory
        use_weighted_sampler: Whether to use weighted sampling for training

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights)
    """
    # Load dataset
    file_paths, labels = load_dataset(dataset_path)

    # Create splits
    train_files, train_labels, val_files, val_labels, test_files, test_labels = \
        create_data_splits(file_paths, labels)

    # Compute class weights for loss function
    class_weights = compute_class_weights(train_labels)

    # Create datasets
    train_dataset = InfantCryDataset(train_files, train_labels, augment=True)
    val_dataset = InfantCryDataset(val_files, val_labels, augment=False)
    test_dataset = InfantCryDataset(test_files, test_labels, augment=False)

    # Create weighted sampler if enabled
    if use_weighted_sampler:
        train_sampler = create_weighted_sampler(train_labels)
        shuffle = False  # Don't shuffle when using sampler
        print("\nUsing weighted random sampler for training")
    else:
        train_sampler = None
        shuffle = True
        print("\nUsing standard random shuffling for training")

    # Create data loaders with custom collate function
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle if train_sampler is None else False,
        collate_fn=collate_fn_wav2vec2,
        num_workers=2,  # Reduced for Wav2Vec2
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_wav2vec2,
        num_workers=2,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_wav2vec2,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, class_weights
