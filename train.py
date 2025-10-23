"""
Training script for Wav2Vec2-based infant cry classification
"""
import os
import sys
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import config
from dataset import get_data_loaders
from model import get_model, print_model_info


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch,
                gradient_accumulation_steps):
    """
    Train for one epoch

    Args:
        model: Wav2Vec2 model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch: Current epoch number
        gradient_accumulation_steps: Steps to accumulate gradients

    Returns:
        Average training loss and accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        input_values = batch['input_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_values, attention_mask=attention_mask)
        loss = criterion(outputs, labels)

        # Normalize loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()

        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Statistics
        running_loss += loss.item() * gradient_accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item() * gradient_accumulation_steps:.4f} '
                  f'Acc: {100.*correct/total:.2f}% '
                  f'LR: {scheduler.get_last_lr()[0]:.2e}')

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    Validate the model

    Args:
        model: Wav2Vec2 model
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
        for batch in val_loader:
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_values, attention_mask=attention_mask)
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
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')

    # Create data loaders
    print('\nLoading dataset...')
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(
        batch_size=args.batch_size,
        dataset_path=args.data_path,
        use_weighted_sampler=args.use_weighted_sampler
    )

    # Create model
    print(f'\nCreating Wav2Vec2 model...')
    print(f'Model: {args.model_name}')
    print(f'Model type: {args.model_type}')

    model = get_model(
        model_type=args.model_type,
        model_name=args.model_name,
        num_labels=config.NUM_CLASSES,
        dropout=config.DROPOUT,
        freeze_feature_extractor=config.FREEZE_FEATURE_EXTRACTOR
    )
    model = model.to(device)

    # Print model info
    print_model_info(model)

    # Loss function and optimizer
    if args.use_class_weights:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("\nUsing weighted CrossEntropyLoss with class weights")
    else:
        criterion = nn.CrossEntropyLoss()
        print("\nUsing standard CrossEntropyLoss")

    # Optimizer - different learning rates for pretrained and new layers
    optimizer = optim.AdamW([
        {'params': model.wav2vec2.parameters(), 'lr': args.learning_rate},
        {'params': model.classifier.parameters(), 'lr': args.learning_rate * 10}
    ], weight_decay=config.WEIGHT_DECAY)

    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * args.epochs // config.GRADIENT_ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=num_training_steps
    )

    # TensorBoard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(config.LOG_DIR, f'wav2vec2_{timestamp}')
    writer = SummaryWriter(log_dir)

    # Training loop
    best_val_acc = 0.0
    best_model_path = os.path.join(config.MODEL_DIR, 'best_wav2vec2.pth')

    print('\nStarting training...')
    for epoch in range(1, args.epochs + 1):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch}/{args.epochs}')
        print(f'{"="*60}')

        # Unfreeze feature extractor after FREEZE_EPOCHS
        if epoch == config.FREEZE_EPOCHS + 1 and config.FREEZE_FEATURE_EXTRACTOR:
            print("\nUnfreezing feature extractor...")
            model.unfreeze_feature_extractor()
            print_model_info(model)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch,
            config.GRADIENT_ACCUMULATION_STEPS
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Print epoch summary
        print(f'\nEpoch {epoch} Summary:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': {
                    'model_name': args.model_name,
                    'model_type': args.model_type,
                    'num_classes': config.NUM_CLASSES
                }
            }, best_model_path)
            print(f'Best model saved with val_acc: {val_acc:.2f}%')

        # Save checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR, f'wav2vec2_epoch_{epoch}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)

    print('\nTraining completed!')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')
    print(f'Best model saved at: {best_model_path}')

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Wav2Vec2 infant cry classification model')
    parser.add_argument('--data-path', type=str, default=config.DATASET_PATH,
                       help='Path to dataset')
    parser.add_argument('--model-name', type=str, default=config.WAV2VEC2_MODEL_NAME,
                       help='Wav2Vec2 model name (e.g., facebook/wav2vec2-base)')
    parser.add_argument('--model-type', type=str, default='standard',
                       choices=['standard', 'attention_pooling'],
                       help='Model type (standard or attention_pooling)')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=config.LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--save-freq', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--use-class-weights', action='store_true',
                       default=config.USE_CLASS_WEIGHTS,
                       help='Use class weights in loss function')
    parser.add_argument('--use-weighted-sampler', action='store_true',
                       default=config.USE_WEIGHTED_SAMPLER,
                       help='Use weighted random sampler for training')
    parser.add_argument('--no-class-weights', dest='use_class_weights',
                       action='store_false',
                       help='Disable class weights')
    parser.add_argument('--no-weighted-sampler', dest='use_weighted_sampler',
                       action='store_false',
                       help='Disable weighted sampler')

    args = parser.parse_args()
    train(args)
