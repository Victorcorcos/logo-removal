#!/usr/bin/env python3
"""
Setup script to download the YOLO model for logo detection.
This script downloads the pre-trained YOLOv11 model for watermark/logo detection.
"""

import os
import sys
import argparse
from pathlib import Path
import urllib.request
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model information
DEFAULT_MODEL_URL = "https://huggingface.co/corzent/yolo11x_watermark_detection/resolve/main/best.pt"
DEFAULT_MODEL_NAME = "best.pt"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = BASE_DIR / "models"


def download_file(url: str, destination: str) -> bool:
    """
    Download a file from URL to destination with progress bar.

    Args:
        url: URL to download from
        destination: Local file path to save to

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading from: {url}")
        logger.info(f"Saving to: {destination}")

        def reporthook(count, block_size, total_size):
            """Progress bar callback."""
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\rProgress: {percent}% ")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, destination, reporthook)
        print()  # New line after progress
        logger.info("Download completed successfully")
        return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def verify_model(model_path: str) -> bool:
    """
    Verify that the model file exists and is valid.

    Args:
        model_path: Path to the model file

    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False

    file_size = os.path.getsize(model_path)
    if file_size < 1024 * 1024:  # Less than 1MB is suspicious
        logger.error(f"Model file seems too small: {file_size} bytes")
        return False

    logger.info(f"Model file verified: {file_size / (1024*1024):.2f} MB")
    return True


def setup_model(model_url: str = DEFAULT_MODEL_URL,
                model_name: str = DEFAULT_MODEL_NAME,
                models_dir: str = str(DEFAULT_MODELS_DIR),
                force: bool = False) -> bool:
    """
    Download and setup the YOLO model.

    Args:
        model_url: URL to download model from
        model_name: Name to save the model as
        models_dir: Directory to save models in
        force: Force re-download even if file exists

    Returns:
        True if successful, False otherwise
    """
    # Create models directory
    models_path = Path(models_dir).expanduser()
    if not models_path.is_absolute():
        models_path = (BASE_DIR / models_path).resolve()
    models_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Models directory: {models_path.absolute()}")

    # Full path to model file
    model_path = models_path / model_name

    # Check if model already exists
    if model_path.exists() and not force:
        logger.info(f"Model already exists: {model_path}")
        if verify_model(str(model_path)):
            logger.info("Model is valid. Use --force to re-download.")
            return True
        else:
            logger.warning("Existing model appears invalid. Re-downloading...")

    # Download the model
    logger.info("Starting model download...")
    logger.info("This may take a few minutes depending on your internet connection.")

    success = download_file(model_url, str(model_path))

    if success:
        # Verify the downloaded model
        if verify_model(str(model_path)):
            logger.info("✓ Model setup complete!")
            logger.info(f"✓ Model location: {model_path.absolute()}")
            return True
        else:
            logger.error("Downloaded model failed verification")
            return False
    else:
        logger.error("Failed to download model")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download and setup the YOLO model for logo detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download the default model
  python setup_models.py

  # Force re-download
  python setup_models.py --force

  # Download from custom URL
  python setup_models.py --url https://example.com/custom_model.pt --name custom.pt

  # Specify custom models directory
  python setup_models.py --dir /path/to/models
        """
    )

    parser.add_argument(
        '--url',
        default=DEFAULT_MODEL_URL,
        help=f'URL to download model from (default: {DEFAULT_MODEL_URL})'
    )

    parser.add_argument(
        '--name',
        default=DEFAULT_MODEL_NAME,
        help=f'Name to save the model as (default: {DEFAULT_MODEL_NAME})'
    )

    parser.add_argument(
        '--dir',
        default=str(DEFAULT_MODELS_DIR),
        help=f'Directory to save models in (default: {DEFAULT_MODELS_DIR})'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if model exists'
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("YOLO Model Setup for Logo Detection")
    logger.info("="*60)

    success = setup_model(
        model_url=args.url,
        model_name=args.name,
        models_dir=args.dir,
        force=args.force
    )

    if success:
        logger.info("\n" + "="*60)
        logger.info("Setup completed successfully!")
        logger.info("You can now run the logo removal script:")
        logger.info(f"  python remove_logos.py -i /path/to/images")
        logger.info("="*60)
        sys.exit(0)
    else:
        logger.error("\n" + "="*60)
        logger.error("Setup failed!")
        logger.error("Please check the error messages above and try again.")
        logger.error("="*60)
        sys.exit(1)


if __name__ == '__main__':
    main()
