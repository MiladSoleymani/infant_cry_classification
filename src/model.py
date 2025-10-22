"""
Deep learning model architectures for infant cry classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class CNNModel(nn.Module):
    """
    Convolutional Neural Network for audio classification
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        """
        Initialize CNN model

        Args:
            num_classes: Number of output classes
        """
        super(CNNModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Calculate the size after convolutions and pooling
        # Input: (batch, 1, 128, ~94) -> After 4 poolings: (batch, 256, 8, 5)
        self.fc1 = nn.Linear(256 * 8 * 5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (batch, 1, height, width)

        Returns:
            Output logits (batch, num_classes)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x


class ResidualBlock(nn.Module):
    """
    Residual block for ResNet-like architecture
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetModel(nn.Module):
    """
    ResNet-like model for audio classification
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        """
        Initialize ResNet model

        Args:
            num_classes: Number of output classes
        """
        super(ResNetModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (batch, 1, height, width)

        Returns:
            Output logits (batch, num_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def get_model(model_name: str = 'cnn', num_classes: int = config.NUM_CLASSES) -> nn.Module:
    """
    Get model by name

    Args:
        model_name: Name of the model ('cnn' or 'resnet')
        num_classes: Number of output classes

    Returns:
        PyTorch model
    """
    if model_name.lower() == 'cnn':
        return CNNModel(num_classes)
    elif model_name.lower() == 'resnet':
        return ResNetModel(num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
