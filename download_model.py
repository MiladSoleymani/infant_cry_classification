"""
Download Wav2Vec2 model for offline use
Run this script once before training in offline environments (e.g., Kaggle)
"""
import os
import sys
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import config


def download_model(model_name=config.WAV2VEC2_MODEL_NAME, cache_dir=config.CACHE_DIR):
    """
    Download Wav2Vec2 model and processor to local cache

    Args:
        model_name: Name of the model to download
        cache_dir: Directory to save the model
    """
    print(f"Downloading Wav2Vec2 model: {model_name}")
    print(f"Cache directory: {cache_dir}")

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    try:
        # Download processor
        print("\nDownloading Wav2Vec2Processor...")
        processor = Wav2Vec2Processor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        processor.save_pretrained(cache_dir)
        print(f"✓ Processor saved to {cache_dir}")

        # Download model
        print("\nDownloading Wav2Vec2Model...")
        model = Wav2Vec2Model.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        model.save_pretrained(cache_dir)
        print(f"✓ Model saved to {cache_dir}")

        print("\n" + "="*60)
        print("SUCCESS! Model downloaded successfully.")
        print("="*60)
        print(f"\nModel location: {cache_dir}")
        print("\nTo use the offline model, update config.py:")
        print(f"  WAV2VEC2_MODEL_NAME = '{cache_dir}'")
        print(f"  USE_CACHE = False")

    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify you have access to huggingface.co")
        print("3. Make sure you have enough disk space")
        print(f"4. Try a different cache directory")
        return False

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Download Wav2Vec2 model for offline use')
    parser.add_argument('--model-name', type=str, default=config.WAV2VEC2_MODEL_NAME,
                       help='Model name (e.g., facebook/wav2vec2-base)')
    parser.add_argument('--cache-dir', type=str, default=config.CACHE_DIR,
                       help='Directory to save the model')

    args = parser.parse_args()

    success = download_model(args.model_name, args.cache_dir)

    if not success:
        sys.exit(1)
