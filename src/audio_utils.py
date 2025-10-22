"""
Audio processing utilities for infant cry classification
"""
import librosa
import numpy as np
import torch
from typing import Tuple
import config


def load_audio(file_path: str, sr: int = config.SAMPLE_RATE,
               duration: float = config.DURATION) -> np.ndarray:
    """
    Load audio file and resample to target sample rate

    Args:
        file_path: Path to audio file
        sr: Target sample rate
        duration: Target duration in seconds

    Returns:
        Audio time series as numpy array
    """
    try:
        # Load audio file
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)

        # Pad or trim to fixed length
        target_length = int(sr * duration)
        if len(audio) < target_length:
            # Pad with zeros if too short
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            # Trim if too long
            audio = audio[:target_length]

        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.zeros(int(sr * duration))


def audio_to_melspectrogram(audio: np.ndarray,
                            sr: int = config.SAMPLE_RATE,
                            n_mels: int = config.N_MELS,
                            n_fft: int = config.N_FFT,
                            hop_length: int = config.HOP_LENGTH) -> np.ndarray:
    """
    Convert audio to mel spectrogram

    Args:
        audio: Audio time series
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length for STFT

    Returns:
        Mel spectrogram as numpy array
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


def preprocess_audio(file_path: str) -> torch.Tensor:
    """
    Complete preprocessing pipeline: load audio and convert to mel spectrogram

    Args:
        file_path: Path to audio file

    Returns:
        Preprocessed mel spectrogram as torch tensor
    """
    # Load audio
    audio = load_audio(file_path)

    # Convert to mel spectrogram
    mel_spec = audio_to_melspectrogram(audio)

    # Normalize
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

    # Convert to tensor and add channel dimension
    mel_spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)

    return mel_spec_tensor


def augment_audio(audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Apply data augmentation to audio

    Args:
        audio: Audio time series
        sr: Sample rate

    Returns:
        Augmented audio
    """
    augmented = audio.copy()

    # Random time shift
    if np.random.random() > 0.5:
        shift = np.random.randint(-sr // 2, sr // 2)
        augmented = np.roll(augmented, shift)

    # Random pitch shift
    if np.random.random() > 0.5:
        n_steps = np.random.randint(-2, 3)
        augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=n_steps)

    # Add random noise
    if np.random.random() > 0.5:
        noise = np.random.randn(len(augmented)) * 0.005
        augmented = augmented + noise

    return augmented
