"""
Training script for infant cry classification
"""
import os
import sys
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import config
from dataset import get_data_loaders
from model import get_model, count_parameters


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch

    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Average training loss and accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} '
                  f'Acc: {100.*correct/total:.2f}%')

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    Validate the model

    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Average validation loss and accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def train(args):
    """
    Main training function

    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create data loaders
    print('\nLoading dataset...')
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        dataset_path=args.data_path
    )

    # Create model
    print(f'\nCreating {args.model} model...')
    model = get_model(args.model, config.NUM_CLASSES)
    model = model.to(device)

    # Print model info
    num_params = count_parameters(model)
    print(f'Number of trainable parameters: {num_params:,}')

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # TensorBoard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(config.LOG_DIR, f'{args.model}_{timestamp}')
    writer = SummaryWriter(log_dir)

    # Training loop
    best_val_acc = 0.0
    best_model_path = os.path.join(config.MODEL_DIR, f'best_{args.model}.pth')

    print('\nStarting training...')
    for epoch in range(1, args.epochs + 1):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch}/{args.epochs}')
        print(f'{"="*60}')

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print epoch summary
        print(f'\nEpoch {epoch} Summary:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, best_model_path)
            print(f'Best model saved with val_acc: {val_acc:.2f}%')

        # Save checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR, f'{args.model}_epoch_{epoch}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)

    print('\nTraining completed!')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')
    print(f'Best model saved at: {best_model_path}')

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train infant cry classification model')
    parser.add_argument('--data-path', type=str, default=config.DATASET_PATH,
                       help='Path to dataset')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet'],
                       help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=config.LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='Save checkpoint every N epochs')

    args = parser.parse_args()
    train(args)
