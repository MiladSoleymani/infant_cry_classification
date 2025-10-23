"""
Wav2Vec2-based model architectures for infant cry classification
"""
import os
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
from typing import Optional
import config


class Wav2Vec2ForAudioClassification(nn.Module):
    """
    Wav2Vec2 model with classification head for infant cry classification
    """

    def __init__(
        self,
        model_name: str = config.WAV2VEC2_MODEL_NAME,
        num_labels: int = config.NUM_CLASSES,
        dropout: float = config.DROPOUT,
        freeze_feature_extractor: bool = config.FREEZE_FEATURE_EXTRACTOR
    ):
        """
        Initialize Wav2Vec2 classification model

        Args:
            model_name: Pre-trained Wav2Vec2 model name
            num_labels: Number of output classes
            dropout: Dropout probability
            freeze_feature_extractor: Whether to freeze CNN feature extractor
        """
        super().__init__()

        # Load pre-trained Wav2Vec2 model
        try:
            if config.USE_CACHE and os.path.exists(config.CACHE_DIR):
                print(f"Loading model from cache: {config.CACHE_DIR}")
                self.wav2vec2 = Wav2Vec2Model.from_pretrained(
                    config.CACHE_DIR,
                    local_files_only=True,
                    attention_dropout=config.ATTENTION_DROPOUT,
                    hidden_dropout=config.HIDDEN_DROPOUT,
                    feat_proj_dropout=dropout,
                    mask_time_prob=0.05,
                    layerdrop=0.05
                )
            else:
                print(f"Downloading model from HuggingFace: {model_name}")
                self.wav2vec2 = Wav2Vec2Model.from_pretrained(
                    model_name,
                    cache_dir=config.CACHE_DIR if config.USE_CACHE else None,
                    attention_dropout=config.ATTENTION_DROPOUT,
                    hidden_dropout=config.HIDDEN_DROPOUT,
                    feat_proj_dropout=dropout,
                    mask_time_prob=0.05,
                    layerdrop=0.05
                )
        except Exception as e:
            print(f"\n✗ Error loading Wav2Vec2 model: {e}")
            print("\nTroubleshooting:")
            print("1. If offline, run: python download_model.py")
            print("2. Check internet connection")
            print(f"3. Verify cache directory: {config.CACHE_DIR}")
            raise

        # Freeze feature extractor (CNN layers) if requested
        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()

        # Get hidden size from config
        hidden_size = self.wav2vec2.config.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )

        self.num_labels = num_labels

    def freeze_feature_extractor(self):
        """Freeze the feature extractor (CNN) parameters"""
        self.wav2vec2.feature_extractor._freeze_parameters()

    def unfreeze_feature_extractor(self):
        """Unfreeze the feature extractor parameters"""
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Forward pass

        Args:
            input_values: Raw audio waveform (batch_size, sequence_length)
            attention_mask: Attention mask (batch_size, sequence_length)

        Returns:
            Logits for classification (batch_size, num_labels)
        """
        # Extract features with Wav2Vec2
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=False
        )

        # Get last hidden state
        hidden_states = outputs.last_hidden_state  # (batch, time, hidden_size)

        # Mean pooling over time dimension
        if attention_mask is not None:
            # Mask out padding tokens before pooling
            attention_mask = attention_mask.unsqueeze(-1)
            hidden_states = hidden_states * attention_mask
            pooled = hidden_states.sum(dim=1) / attention_mask.sum(dim=1)
        else:
            pooled = hidden_states.mean(dim=1)

        # Classification
        logits = self.classifier(pooled)

        return logits


class Wav2Vec2WithAttentionPooling(nn.Module):
    """
    Wav2Vec2 with attention-based pooling for better feature aggregation
    """

    def __init__(
        self,
        model_name: str = config.WAV2VEC2_MODEL_NAME,
        num_labels: int = config.NUM_CLASSES,
        dropout: float = config.DROPOUT,
        freeze_feature_extractor: bool = config.FREEZE_FEATURE_EXTRACTOR
    ):
        super().__init__()

        # Load pre-trained Wav2Vec2
        try:
            if config.USE_CACHE and os.path.exists(config.CACHE_DIR):
                print(f"Loading model from cache: {config.CACHE_DIR}")
                self.wav2vec2 = Wav2Vec2Model.from_pretrained(
                    config.CACHE_DIR,
                    local_files_only=True,
                    attention_dropout=config.ATTENTION_DROPOUT,
                    hidden_dropout=config.HIDDEN_DROPOUT,
                    feat_proj_dropout=dropout
                )
            else:
                print(f"Downloading model from HuggingFace: {model_name}")
                self.wav2vec2 = Wav2Vec2Model.from_pretrained(
                    model_name,
                    cache_dir=config.CACHE_DIR if config.USE_CACHE else None,
                    attention_dropout=config.ATTENTION_DROPOUT,
                    hidden_dropout=config.HIDDEN_DROPOUT,
                    feat_proj_dropout=dropout
                )
        except Exception as e:
            print(f"\n✗ Error loading Wav2Vec2 model: {e}")
            print("\nTroubleshooting:")
            print("1. If offline, run: python download_model.py")
            print("2. Check internet connection")
            print(f"3. Verify cache directory: {config.CACHE_DIR}")
            raise

        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()

        hidden_size = self.wav2vec2.config.hidden_size

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )

        self.num_labels = num_labels

    def freeze_feature_extractor(self):
        """Freeze the feature extractor parameters"""
        self.wav2vec2.feature_extractor._freeze_parameters()

    def unfreeze_feature_extractor(self):
        """Unfreeze the feature extractor parameters"""
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """Forward pass with attention pooling"""
        # Extract features
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state  # (batch, time, hidden)

        # Attention pooling
        attention_weights = self.attention(hidden_states)  # (batch, time, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Weighted sum
        pooled = (hidden_states * attention_weights).sum(dim=1)  # (batch, hidden)

        # Classification
        logits = self.classifier(pooled)

        return logits


def get_model(
    model_type: str = "standard",
    model_name: str = config.WAV2VEC2_MODEL_NAME,
    num_labels: int = config.NUM_CLASSES,
    **kwargs
) -> nn.Module:
    """
    Get Wav2Vec2 model by type

    Args:
        model_type: Type of model ("standard" or "attention_pooling")
        model_name: Pre-trained Wav2Vec2 model name
        num_labels: Number of output classes
        **kwargs: Additional arguments

    Returns:
        Wav2Vec2 model
    """
    if model_type == "standard":
        return Wav2Vec2ForAudioClassification(
            model_name=model_name,
            num_labels=num_labels,
            **kwargs
        )
    elif model_type == "attention_pooling":
        return Wav2Vec2WithAttentionPooling(
            model_name=model_name,
            num_labels=num_labels,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters

    Args:
        model: PyTorch model
        trainable_only: Count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_info(model: nn.Module):
    """
    Print model information

    Args:
        model: PyTorch model
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    frozen_params = total_params - trainable_params

    print(f"\nModel Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {frozen_params:,}")
    print(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")
