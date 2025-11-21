#!/usr/bin/env python3
"""
Automatic Logo Removal Tool
Detects and removes logos/watermarks from images using AI-based detection and inpainting.
"""

import os
import sys
import argparse
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, List, Optional
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from transformers import Owlv2VisionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('log/logo_removal.log')
    ]
)
logger = logging.getLogger(__name__)


class DetectorModelOwl(nn.Module):
    """OWLv2-based watermark classifier for enhanced detection."""

    def __init__(self, model_path: str, dropout: float, n_hidden: int = 768):
        super().__init__()

        owl = Owlv2VisionModel.from_pretrained(model_path)
        assert isinstance(owl, Owlv2VisionModel)
        self.owl = owl
        self.owl.requires_grad_(False)

        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(n_hidden, eps=1e-5)
        self.linear1 = nn.Linear(n_hidden, n_hidden * 2)
        self.act1 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(n_hidden * 2, eps=1e-5)
        self.linear2 = nn.Linear(n_hidden * 2, 2)

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            # Embed the image
            outputs = self.owl(pixel_values=pixel_values, output_hidden_states=True)
            x = outputs.last_hidden_state  # B, N, C

            # Linear
            x = self.dropout1(x)
            x = self.ln1(x)
            x = self.linear1(x)
            x = self.act1(x)

            # Norm and Mean
            x = self.dropout2(x)
            x, _ = x.max(dim=1)
            x = self.ln2(x)

            # Linear
            x = self.linear2(x)

        if labels is not None:
            loss = F.cross_entropy(x, labels)
            return (x, loss)

        return (x,)


class LogoRemover:
    """Main class for detecting and removing logos from images."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.25,
                 mask_expansion: int = 15, device: str = 'auto',
                 use_owl: bool = True, owl_weights_path: str = 'models/far5y1y5-8000.pt',
                 inpaint_model: str = 'mat'):
        """
        Initialize the logo remover.

        Args:
            model_path: Path to the YOLO model weights file
            confidence_threshold: Minimum confidence for logo detection (0-1)
            mask_expansion: Pixels to expand the mask on each side
            device: Device to use ('auto', 'cpu', 'cuda', or specific GPU index)
            use_owl: Whether to use OWLv2 classifier for enhanced detection
            owl_weights_path: Path to OWLv2 classifier weights
            inpaint_model: Inpainting model to use ('mat' for anime, 'lama' for photos)
        """
        self.confidence_threshold = confidence_threshold
        self.mask_expansion = mask_expansion
        self.use_owl = use_owl
        self.owl_classifier = None
        self.inpaint_model = inpaint_model
        self.device = device

        # Initialize OWLv2 classifier if enabled
        if self.use_owl and os.path.exists(owl_weights_path):
            logger.info("Loading OWLv2 watermark classifier...")
            try:
                self.owl_classifier = DetectorModelOwl("google/owlv2-base-patch16-ensemble", dropout=0.0)
                self.owl_classifier.load_state_dict(torch.load(owl_weights_path, map_location="cpu"))
                self.owl_classifier.eval()
                logger.info("OWLv2 classifier loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load OWLv2 classifier: {e}")
                logger.warning("Continuing with YOLO detection only")
                self.owl_classifier = None

        # Initialize YOLO detector
        logger.info(f"Loading YOLO model from {model_path}...")
        try:
            self.detector = YOLO(model_path)
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

        # IOPaint inpainting (MAT or LaMa) - no pre-loading needed
        logger.info(f"Using IOPaint with {self.inpaint_model.upper()} model for inpainting")

        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def has_watermark_owl(self, image: Image.Image) -> bool:
        """
        Check if image has a watermark using OWLv2 classifier.

        Args:
            image: PIL Image

        Returns:
            True if watermark detected, False otherwise
        """
        if self.owl_classifier is None:
            return False

        try:
            # Pad to square
            big_side = max(image.size)
            new_image = Image.new("RGB", (big_side, big_side), (128, 128, 128))
            new_image.paste(image, (0, 0))

            # Resize to 960x960
            preped = new_image.resize((960, 960), Image.BICUBIC)

            # Convert to tensor and normalize
            preped = TVF.pil_to_tensor(preped)
            preped = preped / 255.0
            input_image = TVF.normalize(preped, [0.48145466, 0.4578275, 0.40821073],
                                       [0.26862954, 0.26130258, 0.27577711])

            # Run prediction
            with torch.no_grad():
                logits, = self.owl_classifier(input_image.unsqueeze(0), None)
                probs = F.softmax(logits, dim=1)
                prediction = torch.argmax(probs.cpu(), dim=1)

            has_watermark = prediction.item() == 1
            confidence = probs[0][1].item()

            logger.debug(f"OWLv2 watermark prediction: {has_watermark} (confidence: {confidence:.2f})")
            return has_watermark

        except Exception as e:
            logger.warning(f"OWLv2 prediction failed: {e}")
            return False

    def detect_logos(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Detect logos in an image using YOLO.

        Args:
            image_path: Path to the input image

        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        logger.debug(f"Detecting logos in {image_path}")

        try:
            # Run detection with augmentation for better results
            results = self.detector.predict(
                source=image_path,
                conf=self.confidence_threshold,
                imgsz=1024,
                augment=True,
                iou=0.5,
                verbose=False
            )

            # Extract bounding boxes
            boxes = []
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    # Get coordinates in xyxy format
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))
                    logger.debug(f"Found logo at ({x1}, {y1}, {x2}, {y2}) "
                               f"with confidence {box.conf[0]:.2f}")

            logger.info(f"Detected {len(boxes)} logo(s)")
            return boxes

        except Exception as e:
            logger.error(f"Error during logo detection: {e}")
            return []

    def create_mask(self, image_shape: Tuple[int, int],
                   boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Create a binary mask for the logo regions with expansion.

        Args:
            image_shape: Shape of the image (height, width)
            boxes: List of bounding boxes [(x1, y1, x2, y2), ...]

        Returns:
            Binary mask as numpy array (255 for logo, 0 for background)
        """
        height, width = image_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        for x1, y1, x2, y2 in boxes:
            # Expand the bounding box
            x1_exp = max(0, x1 - self.mask_expansion)
            y1_exp = max(0, y1 - self.mask_expansion)
            x2_exp = min(width, x2 + self.mask_expansion)
            y2_exp = min(height, y2 + self.mask_expansion)

            # Fill the mask region with white
            mask[y1_exp:y2_exp, x1_exp:x2_exp] = 255

            logger.debug(f"Created mask region: ({x1_exp}, {y1_exp}, {x2_exp}, {y2_exp})")

        return mask

    def inpaint_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Remove logo using IOPaint (MAT or LaMa).

        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            mask: Binary mask (255 for areas to inpaint, 0 otherwise)

        Returns:
            Inpainted image as numpy array
        """
        logger.debug(f"Starting inpainting process with {self.inpaint_model.upper()} model")

        try:
            # Create temporary files for input/output
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_image_path = os.path.join(tmpdir, 'input.png')
                tmp_mask_path = os.path.join(tmpdir, 'mask.png')
                tmp_output_dir = os.path.join(tmpdir, 'output')

                # Save image and mask
                cv2.imwrite(tmp_image_path, image)
                cv2.imwrite(tmp_mask_path, mask)

                # Run IOPaint command
                device_param = 'cuda' if self.device != 'cpu' and torch.cuda.is_available() else 'cpu'
                cmd = [
                    'iopaint', 'run',
                    '--model', self.inpaint_model,
                    '--device', device_param,
                    '--image', tmp_image_path,
                    '--mask', tmp_mask_path,
                    '--output', tmp_output_dir
                ]

                logger.debug(f"Running IOPaint: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)

                # IOPaint creates a subdirectory and saves output with same name as input
                result_image_path = os.path.join(tmp_output_dir, 'input.png')
                result_image = cv2.imread(result_image_path)

                if result_image is None:
                    raise Exception("IOPaint failed to generate output")

                logger.debug("Inpainting completed successfully")
                return result_image

        except subprocess.CalledProcessError as e:
            logger.error(f"IOPaint command failed: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Error during inpainting: {e}")
            raise

    def process_image(self, input_path: str, output_path: str) -> bool:
        """
        Process a single image: detect, mask, and remove logos.

        Args:
            input_path: Path to input image
            output_path: Path to save processed image

        Returns:
            True if successful, False otherwise
        """
        try:
            # Read the image
            logger.info(f"Processing: {input_path}")
            image = cv2.imread(input_path)

            if image is None:
                logger.error(f"Failed to read image: {input_path}")
                return False

            # Convert to PIL for OWLv2 check
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            # Use OWLv2 to pre-check for watermarks if available
            if self.owl_classifier is not None:
                has_watermark = self.has_watermark_owl(pil_image)
                if not has_watermark:
                    logger.info(f"OWLv2 classifier: No watermark detected in {input_path}")
                    logger.info("Skipping YOLO detection and saving original image.")
                    cv2.imwrite(output_path, image)
                    return True
                else:
                    logger.info("OWLv2 classifier: Watermark detected, proceeding with YOLO localization")

            # Detect logos with YOLO
            boxes = self.detect_logos(input_path)

            if len(boxes) == 0:
                if self.owl_classifier is not None and has_watermark:
                    logger.warning(f"OWLv2 detected watermark but YOLO found no boxes in {input_path}")
                    logger.warning("Cannot remove watermark without location - saving original image.")
                else:
                    logger.warning(f"No logos detected by YOLO in {input_path}.")
                    logger.warning("Saving original image.")

                # Save original image to output
                cv2.imwrite(output_path, image)
                return True

            # Create mask
            mask = self.create_mask(image.shape, boxes)

            # Inpaint to remove logos
            result = self.inpaint_image(image, mask)

            # Save result
            cv2.imwrite(output_path, result)
            logger.info(f"Successfully saved to: {output_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to process {input_path}: {e}")
            return False

    def process_directory(self, input_dir: str, output_dir: str = None) -> dict:
        """
        Process all images in a directory.

        Args:
            input_dir: Path to input directory
            output_dir: Path to output directory (default: input_dir/no_logos)

        Returns:
            Dictionary with processing statistics
        """
        input_path = Path(input_dir)

        if not input_path.exists() or not input_path.is_dir():
            logger.error(f"Input directory does not exist: {input_dir}")
            return {'success': 0, 'failed': 0, 'skipped': 0}

        # Set output directory
        if output_dir is None:
            output_path = input_path / 'no_logos'
        else:
            output_path = Path(output_dir)

        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_path}")

        # Find all images
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))

        total = len(image_files)
        logger.info(f"Found {total} image(s) to process")

        # Process statistics
        stats = {'success': 0, 'failed': 0, 'skipped': 0}

        # Process each image
        for idx, img_path in enumerate(image_files, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Image {idx}/{total}")
            logger.info(f"{'='*60}")

            output_file = output_path / img_path.name

            # Skip if already processed
            if output_file.exists():
                logger.info(f"Skipping (already exists): {img_path.name}")
                stats['skipped'] += 1
                continue

            # Process the image
            success = self.process_image(str(img_path), str(output_file))

            if success:
                stats['success'] += 1
            else:
                stats['failed'] += 1

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("PROCESSING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total images: {total}")
        logger.info(f"Successfully processed: {stats['success']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Skipped (already exists): {stats['skipped']}")
        logger.info(f"{'='*60}\n")

        return stats


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Automatically detect and remove logos from images using AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images in a directory
  python remove_logos.py -i /path/to/images

  # Specify custom output directory
  python remove_logos.py -i /path/to/images -o /path/to/output

  # Use custom YOLO model
  python remove_logos.py -i /path/to/images -m custom_model.pt

  # Adjust detection sensitivity
  python remove_logos.py -i /path/to/images -c 0.5 -e 20
        """
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input directory containing images with logos'
    )

    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output directory (default: INPUT_DIR/no_logos)'
    )

    parser.add_argument(
        '-m', '--model',
        default='models/best.pt',
        help='Path to YOLO model weights file (default: models/best.pt)'
    )

    parser.add_argument(
        '-c', '--confidence',
        type=float,
        default=0.25,
        help='Confidence threshold for detection (0-1, default: 0.25)'
    )

    parser.add_argument(
        '-e', '--expansion',
        type=int,
        default=15,
        help='Mask expansion in pixels (default: 15)'
    )

    parser.add_argument(
        '-d', '--device',
        default='auto',
        help='Device to use: auto, cpu, cuda, or GPU index (default: auto)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--inpaint-model',
        default='mat',
        choices=['mat', 'lama'],
        help='Inpainting model: mat (better for anime) or lama (better for photos, default: mat)'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Check if model file exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        logger.error("Please download the model first. See README.md for instructions.")
        sys.exit(1)

    try:
        # Initialize remover
        remover = LogoRemover(
            model_path=args.model,
            confidence_threshold=args.confidence,
            mask_expansion=args.expansion,
            device=args.device,
            inpaint_model=args.inpaint_model
        )

        # Process directory
        stats = remover.process_directory(args.input, args.output)

        # Exit with appropriate code
        if stats['failed'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
