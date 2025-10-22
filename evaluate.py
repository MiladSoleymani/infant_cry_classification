"""
Evaluation script for infant cry classification
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import config
from dataset import get_data_loaders
from model import get_model


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set

    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on

    Returns:
        Tuple of (predictions, true_labels, probabilities)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)

            # Get predictions
            _, predicted = outputs.max(1)

            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Confusion matrix saved to: {save_path}')

    plt.show()


def plot_class_accuracies(cm, class_names, save_path=None):
    """
    Plot per-class accuracies

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    # Calculate per-class accuracy
    class_accuracies = cm.diagonal() / cm.sum(axis=1) * 100

    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, class_accuracies, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy')
    plt.ylim([0, 100])
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Class accuracies plot saved to: {save_path}')

    plt.show()


def evaluate(args):
    """
    Main evaluation function

    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create data loaders
    print('\nLoading dataset...')
    _, _, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        dataset_path=args.data_path
    )

    # Create model
    print(f'\nCreating {args.model} model...')
    model = get_model(args.model, config.NUM_CLASSES)
    model = model.to(device)

    # Load trained weights
    print(f'\nLoading model from: {args.model_path}')
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Model loaded from epoch {checkpoint.get("epoch", "unknown")}')
    print(f'Validation accuracy: {checkpoint.get("val_acc", "unknown")}%')

    # Evaluate
    print('\nEvaluating on test set...')
    predictions, true_labels, probabilities = evaluate_model(model, test_loader, device)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)

    print(f'\n{"="*60}')
    print(f'Test Set Evaluation Results')
    print(f'{"="*60}')
    print(f'Overall Accuracy: {accuracy*100:.2f}%')
    print(f'\nClassification Report:')
    print(classification_report(true_labels, predictions,
                                target_names=config.CLASSES,
                                digits=4))

    # Per-class accuracy
    print('\nPer-Class Accuracy:')
    for i, class_name in enumerate(config.CLASSES):
        class_acc = cm[i, i] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
        print(f'  {class_name}: {class_acc:.2f}%')

    # Print confusion matrix
    print(f'\nConfusion Matrix:')
    print(cm)

    # Plot confusion matrix
    if args.plot:
        cm_path = os.path.join(config.MODEL_DIR, 'confusion_matrix.png')
        plot_confusion_matrix(cm, config.CLASSES, cm_path)

        # Plot per-class accuracies
        acc_path = os.path.join(config.MODEL_DIR, 'class_accuracies.png')
        plot_class_accuracies(cm, config.CLASSES, acc_path)

    # Save results
    if args.save_results:
        results_path = os.path.join(config.MODEL_DIR, 'evaluation_results.txt')
        with open(results_path, 'w') as f:
            f.write(f'Test Set Evaluation Results\n')
            f.write(f'{"="*60}\n')
            f.write(f'Overall Accuracy: {accuracy*100:.2f}%\n\n')
            f.write(f'Classification Report:\n')
            f.write(classification_report(true_labels, predictions,
                                         target_names=config.CLASSES,
                                         digits=4))
            f.write(f'\n\nConfusion Matrix:\n')
            f.write(str(cm))

        print(f'\nResults saved to: {results_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate infant cry classification model')
    parser.add_argument('--data-path', type=str, default=config.DATASET_PATH,
                       help='Path to dataset')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet'],
                       help='Model architecture')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--plot', action='store_true',
                       help='Plot confusion matrix and class accuracies')
    parser.add_argument('--save-results', action='store_true',
                       help='Save evaluation results to file')

    args = parser.parse_args()
    evaluate(args)
