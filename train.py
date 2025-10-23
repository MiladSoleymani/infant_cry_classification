"""
Training script for Wav2Vec2-based infant cry classification
Using HuggingFace Trainer and CTCTrainer (based on working notebook implementation)
"""
import os
import sys
import argparse
from typing import Any, Dict, Union

import torch

from transformers import (
    AutoConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import config
from src.dataset import (
    load_and_prepare_datasets,
    DataCollatorCTCWithPadding,
    compute_metrics
)
from src.model import Wav2Vec2ForSpeechClassification, print_model_info


# We can use the standard Trainer since our model returns loss in forward pass
# CTCTrainer was needed in the original notebook for specific use cases
# but our Wav2Vec2ForSpeechClassification already handles loss computation


def train(
    dataset_path: str = config.DATASET_PATH,
    output_dir: str = config.MODEL_DIR,
    num_train_epochs: int = config.EPOCHS,
    per_device_train_batch_size: int = config.BATCH_SIZE,
    gradient_accumulation_steps: int = config.GRADIENT_ACCUMULATION_STEPS,
    learning_rate: float = config.LEARNING_RATE,
    warmup_steps: int = config.WARMUP_STEPS,
    weight_decay: float = config.WEIGHT_DECAY,
    logging_steps: int = 10,
    eval_steps: int = 100,
    save_steps: int = 100,
    save_total_limit: int = 2,
    fp16: bool = True,
    seed: int = config.RANDOM_SEED,
):
    """
    Train Wav2Vec2 model for infant cry classification

    Args:
        dataset_path: Path to dataset directory
        output_dir: Directory to save model checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Number of gradient accumulation steps
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay
        logging_steps: Log every N steps
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        save_total_limit: Maximum number of checkpoints to keep
        fp16: Whether to use mixed precision training
        seed: Random seed
    """
    # Set seed for reproducibility
    set_seed(seed)

    print("="*80)
    print("Wav2Vec2 Infant Cry Classification Training")
    print("="*80)

    # Load and prepare datasets
    print("\nLoading and preparing datasets...")
    if config.USE_UNDERSAMPLING:
        print(f"⚖️  Undersampling enabled: strategy = '{config.UNDERSAMPLING_STRATEGY}'")
    if config.USE_OVERSAMPLING:
        print(f"📈 Oversampling with augmentation enabled: strategy = '{config.OVERSAMPLING_STRATEGY}'")
    # Use current directory for CSV files if output_dir has no parent
    csv_save_path = os.path.dirname(output_dir) if os.path.dirname(output_dir) else "."
    train_dataset, eval_dataset, test_dataset, feature_extractor, label_list, num_labels = \
        load_and_prepare_datasets(
            dataset_path=dataset_path,
            save_path=csv_save_path,
            model_name=config.WAV2VEC2_MODEL_NAME,
            use_undersampling=config.USE_UNDERSAMPLING,
            undersampling_strategy=config.UNDERSAMPLING_STRATEGY,
            use_oversampling=config.USE_OVERSAMPLING,
            oversampling_strategy=config.OVERSAMPLING_STRATEGY
        )

    print(f"\nDataset loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(eval_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Number of classes: {num_labels}")
    print(f"  Classes: {label_list}")

    # Create label mappings
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    # Create model configuration
    print(f"\nCreating model configuration...")
    model_config = AutoConfig.from_pretrained(
        config.WAV2VEC2_MODEL_NAME,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        finetuning_task="wav2vec2_clf",
    )

    # Add custom config parameters
    setattr(model_config, 'pooling_mode', config.POOLING_MODE)
    setattr(model_config, 'final_dropout', config.FINAL_DROPOUT)

    # Create model
    print(f"\nLoading pre-trained Wav2Vec2 model: {config.WAV2VEC2_MODEL_NAME}")
    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        config.WAV2VEC2_MODEL_NAME,
        config=model_config,
    )

    # Freeze feature extractor if specified
    if config.FREEZE_FEATURE_EXTRACTOR:
        print("\nFreezing feature extractor (CNN layers)")
        model.freeze_feature_extractor()

    # Print model info
    print_model_info(model)

    # Create data collator
    data_collator = DataCollatorCTCWithPadding(
        feature_extractor=feature_extractor,
        padding=True
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # evaluation_strategy="steps",
        num_train_epochs=num_train_epochs,
        fp16=fp16,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        save_total_limit=save_total_limit,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
        remove_unused_columns=True,
        report_to=['tensorboard'],
        seed=seed,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # tokenizer=feature_extractor,
    )

    # Train
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")

    torch.cuda.empty_cache()
    train_result = trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model()

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluate on validation set
    print("\n" + "="*80)
    print("Evaluating on validation set...")
    print("="*80 + "\n")

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print(f"\nValidation Results:")
    print(f"  Accuracy: {eval_metrics['eval_accuracy']:.4f}")
    print(f"  Loss: {eval_metrics['eval_loss']:.4f}")

    # Generate detailed classification metrics
    print("\n" + "="*80)
    print("Computing detailed metrics (confusion matrix, precision, recall, F1)...")
    print("="*80 + "\n")

    # Get predictions on validation set
    predictions = trainer.predict(eval_dataset)
    pred_labels = predictions.predictions.argmax(-1)
    true_labels = predictions.label_ids

    # Compute classification metrics
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        precision_recall_fscore_support,
    )
    import numpy as np

    # Classification report
    print("Classification Report:")
    print("-" * 80)
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=label_list,
        digits=4
    )
    print(report)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, average=None, labels=range(num_labels)
    )

    print("\nPer-Class Metrics:")
    print("-" * 80)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    for i, class_name in enumerate(label_list):
        print(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")

    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='weighted'
    )

    print("-" * 80)
    print(f"{'Macro Avg':<15} {precision_macro:<12.4f} {recall_macro:<12.4f} {f1_macro:<12.4f}")
    print(f"{'Weighted Avg':<15} {precision_weighted:<12.4f} {recall_weighted:<12.4f} {f1_weighted:<12.4f}")
    print("-" * 80)

    # Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    print("\nConfusion Matrix:")
    print("-" * 80)

    # Print header
    header = "True\\Pred".ljust(15)
    for class_name in label_list:
        header += f"{class_name[:10]:<12}"
    print(header)
    print("-" * 80)

    # Print matrix with row labels
    for i, class_name in enumerate(label_list):
        row = f"{class_name:<15}"
        for j in range(len(label_list)):
            row += f"{cm[i][j]:<12}"
        print(row)
    print("-" * 80)

    # Save confusion matrix plot
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=label_list,
            yticklabels=label_list,
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix - Validation Set', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Confusion matrix plot saved to: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"\n⚠ Could not save confusion matrix plot: {e}")

    # Save metrics to JSON file
    import json
    detailed_metrics = {
        'accuracy': float(eval_metrics['eval_accuracy']),
        'loss': float(eval_metrics['eval_loss']),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'per_class_metrics': {
            label_list[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
            for i in range(num_labels)
        },
        'confusion_matrix': cm.tolist()
    }

    metrics_path = os.path.join(output_dir, 'detailed_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    print(f"✓ Detailed metrics saved to: {metrics_path}")

    print("\n" + "="*80)
    print("Training completed!")
    print(f"Model saved to: {output_dir}")
    print("="*80)

    return trainer


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Wav2Vec2 for infant cry classification')

    parser.add_argument('--dataset_path', type=str, default=config.DATASET_PATH,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default=config.MODEL_DIR,
                        help='Output directory for model checkpoints')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch size per device')
    parser.add_argument('--learning_rate', type=float, default=config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED,
                        help='Random seed')
    parser.add_argument('--no_fp16', action='store_true',
                        help='Disable mixed precision training')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Train model
    trainer = train(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fp16=not args.no_fp16,
        seed=args.seed,
    )

    return trainer


if __name__ == "__main__":
    main()
