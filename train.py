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
    # Use current directory for CSV files if output_dir has no parent
    csv_save_path = os.path.dirname(output_dir) if os.path.dirname(output_dir) else "."
    train_dataset, eval_dataset, test_dataset, feature_extractor, label_list, num_labels = \
        load_and_prepare_datasets(
            dataset_path=dataset_path,
            save_path=csv_save_path,
            model_name=config.WAV2VEC2_MODEL_NAME
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
