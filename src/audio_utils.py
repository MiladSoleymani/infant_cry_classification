"""
Audio processing utilities for Wav2Vec2-based infant cry classification
"""
import librosa
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor
from typing import Tuple, Optional
import config


# Initialize Wav2Vec2 processor globally
processor = None


def get_processor():
    """
    Get or initialize Wav2Vec2 processor

    Returns:
        Wav2Vec2Processor instance
    """
    global processor
    if processor is None:
        processor = Wav2Vec2Processor.from_pretrained(config.WAV2VEC2_MODEL_NAME)
    return processor


def load_audio(file_path: str, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Load audio file and resample to target sample rate

    Args:
        file_path: Path to audio file
        sr: Target sample rate (16000 for Wav2Vec2)

    Returns:
        Audio array
    """
    try:
        # Load audio using librosa
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.zeros(sr)


def pad_or_truncate(audio: np.ndarray, target_length: int = config.TARGET_LENGTH) -> np.ndarray:
    """
    Pad or truncate audio to fixed length

    Args:
        audio: Audio array
        target_length: Target length in samples

    Returns:
        Padded or truncated audio
    """
    if len(audio) > target_length:
        # Truncate
        audio = audio[:target_length]
    elif len(audio) < target_length:
        # Pad with zeros
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')

    return audio


def augment_audio(audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Apply data augmentation to audio

    Args:
        audio: Audio array
        sr: Sample rate

    Returns:
        Augmented audio
    """
    augmented = audio.copy()

    # Random time shift
    if np.random.random() > 0.5:
        shift = np.random.randint(-sr // 4, sr // 4)
        augmented = np.roll(augmented, shift)

    # Random pitch shift (smaller range for Wav2Vec2)
    if np.random.random() > 0.5:
        n_steps = np.random.randint(-1, 2)
        if n_steps != 0:
            augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=n_steps)

    # Add random noise
    if np.random.random() > 0.5:
        noise = np.random.randn(len(augmented)) * 0.003
        augmented = augmented + noise

    # Random gain
    if np.random.random() > 0.5:
        gain = np.random.uniform(0.8, 1.2)
        augmented = augmented * gain

    # Clip to prevent overflow
    augmented = np.clip(augmented, -1.0, 1.0)

    return augmented


def preprocess_audio_for_wav2vec2(
    file_path: str,
    augment: bool = False,
    return_tensors: str = "pt"
) -> dict:
    """
    Complete preprocessing pipeline for Wav2Vec2

    Args:
        file_path: Path to audio file
        augment: Whether to apply augmentation
        return_tensors: Type of tensors to return ("pt" for PyTorch)

    Returns:
        Dictionary with input_values and attention_mask
    """
    # Load audio
    audio = load_audio(file_path)

    # Apply augmentation if requested
    if augment:
        audio = augment_audio(audio)

    # Pad or truncate
    audio = pad_or_truncate(audio)

    # Process with Wav2Vec2Processor
    # This normalizes and converts to the format expected by Wav2Vec2
    proc = get_processor()
    inputs = proc(
        audio,
        sampling_rate=config.SAMPLE_RATE,
        return_tensors=return_tensors,
        padding=True,
        return_attention_mask=True
    )

    return inputs


def batch_preprocess_audio(
    audio_arrays: list,
    return_tensors: str = "pt"
) -> dict:
    """
    Batch preprocess multiple audio arrays

    Args:
        audio_arrays: List of audio arrays
        return_tensors: Type of tensors to return

    Returns:
        Dictionary with batched input_values and attention_mask
    """
    proc = get_processor()

    # Pad or truncate all audios
    processed_audios = [pad_or_truncate(audio) for audio in audio_arrays]

    # Batch process
    inputs = proc(
        processed_audios,
        sampling_rate=config.SAMPLE_RATE,
        return_tensors=return_tensors,
        padding=True,
        return_attention_mask=True
    )

    return inputs


def compute_audio_length(file_path: str) -> float:
    """
    Compute duration of audio file in seconds

    Args:
        file_path: Path to audio file

    Returns:
        Duration in seconds
    """
    try:
        audio = load_audio(file_path)
        return len(audio) / config.SAMPLE_RATE
    except:
        return 0.0
