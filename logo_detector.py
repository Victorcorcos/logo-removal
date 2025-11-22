#!/usr/bin/env python3
"""
Logo Detection Module
Handles all logo/watermark detection logic using YOLO and OWLv2.
"""

import os
import tempfile
import logging
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from transformers import Owlv2VisionModel
from ultralytics import YOLO

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


class LogoDetector:
    """Handles logo detection using YOLO with optional OWLv2 pre-classification."""

    def __init__(self, yolo_model_path: str, confidence_threshold: float = 0.15,
                 use_owl: bool = True, owl_weights_path: str = 'models/far5y1y5-8000.pt',
                 owl_threshold: float = 0.40):
        """
        Initialize the logo detector.

        Args:
            yolo_model_path: Path to the YOLO model weights file
            confidence_threshold: Minimum confidence for YOLO detection (0-1)
            use_owl: Whether to use OWLv2 classifier for pre-detection
            owl_weights_path: Path to OWLv2 classifier weights
            owl_threshold: Threshold for OWLv2 watermark classification (0-1)
        """
        self.confidence_threshold = confidence_threshold
        self.use_owl = use_owl
        self.owl_threshold = owl_threshold
        self.owl_classifier = None

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
        logger.info(f"Loading YOLO model from {yolo_model_path}...")
        try:
            self.yolo_model = YOLO(yolo_model_path)
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def has_watermark_owl(self, image: Image.Image) -> Tuple[bool, float]:
        """
        Check if image has a watermark using OWLv2 classifier.

        Args:
            image: PIL Image

        Returns:
            Tuple of (has_watermark, confidence_score)
        """
        if self.owl_classifier is None:
            return False, 0.0

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

            # Use configurable threshold
            return confidence >= self.owl_threshold, confidence

        except Exception as e:
            logger.warning(f"OWLv2 prediction failed: {e}")
            return False, 0.0

    def _detect_at_angle(self, image_path: str, angle: float) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect logos at a specific rotation angle.

        Args:
            image_path: Path to the input image
            angle: Rotation angle in degrees (0 for original orientation)

        Returns:
            List of bounding boxes with confidence scores [(x1, y1, x2, y2, conf), ...]
        """
        boxes = []
        temp_file = None

        try:
            img = cv2.imread(image_path)
            if img is None:
                return boxes

            h, w = img.shape[:2]
            current_path = image_path

            # Rotate image if angle is not 0
            if angle != 0:
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)

                # Calculate new image size to avoid cropping
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_w = int((h * sin) + (w * cos))
                new_h = int((h * cos) + (w * sin))

                # Adjust rotation matrix for new center
                M[0, 2] += (new_w / 2) - center[0]
                M[1, 2] += (new_h / 2) - center[1]

                # Rotate image
                rotated = cv2.warpAffine(img, M, (new_w, new_h),
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))

                # Save to temp file
                temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                cv2.imwrite(temp_file.name, rotated)
                current_path = temp_file.name
                temp_file.close()

                logger.debug(f"Testing with {angle}° rotation")

            # Run YOLO detection
            results = self.yolo_model.predict(
                source=current_path,
                conf=self.confidence_threshold,
                imgsz=1024,
                augment=True,
                iou=0.5,
                verbose=False
            )

            # Extract and transform boxes back to original coordinates
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].item()

                    # Transform coordinates back if image was rotated
                    if angle != 0:
                        # Calculate inverse rotation matrix
                        M_inv = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), -angle, 1.0)
                        M_inv[0, 2] += center[0] - (new_w / 2)
                        M_inv[1, 2] += center[1] - (new_h / 2)

                        # Transform corners of bounding box
                        corners = np.array([
                            [x1, y1, 1],
                            [x2, y1, 1],
                            [x1, y2, 1],
                            [x2, y2, 1]
                        ]).T

                        transformed = M_inv @ corners

                        # Get new bounding box from transformed corners
                        x1 = int(np.clip(np.min(transformed[0]), 0, w))
                        y1 = int(np.clip(np.min(transformed[1]), 0, h))
                        x2 = int(np.clip(np.max(transformed[0]), 0, w))
                        y2 = int(np.clip(np.max(transformed[1]), 0, h))
                    else:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    boxes.append((x1, y1, x2, y2, conf))
                    logger.debug(f"Found logo at ({x1}, {y1}, {x2}, {y2}) "
                               f"with confidence {conf:.2f} (angle: {angle}°)")

        except Exception as e:
            logger.error(f"Error detecting at angle {angle}°: {e}")

        finally:
            # Clean up temp file
            if temp_file is not None and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

        return boxes

    def _nms_boxes(self, boxes: List[Tuple[int, int, int, int, float]],
                   iou_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections.

        Args:
            boxes: List of boxes with confidence scores [(x1, y1, x2, y2, conf), ...]
            iou_threshold: IoU threshold for considering boxes as duplicates

        Returns:
            List of filtered boxes [(x1, y1, x2, y2), ...]
        """
        if len(boxes) == 0:
            return []

        # Convert to numpy array
        boxes_array = np.array(boxes)
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 2]
        y2 = boxes_array[:, 3]
        scores = boxes_array[:, 4]

        # Calculate areas
        areas = (x2 - x1) * (y2 - y1)

        # Sort by confidence score
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h

            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            # Keep boxes with IoU below threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        # Return filtered boxes without confidence scores
        result = [(int(boxes_array[i, 0]), int(boxes_array[i, 1]),
                  int(boxes_array[i, 2]), int(boxes_array[i, 3]))
                 for i in keep]

        return result

    def detect(self, image_path: str, use_multiangle: bool = True) -> List[Tuple[int, int, int, int]]:
        """
        Detect logos in an image.

        Args:
            image_path: Path to the input image
            use_multiangle: Whether to use multi-angle detection for diagonal watermarks

        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        logger.debug(f"Detecting logos in {image_path}")

        try:
            all_boxes = []

            if use_multiangle:
                # Test with original image and slight rotations (±15°) to catch diagonal watermarks
                angles = [0, -15, 15]
            else:
                # Only test original orientation
                angles = [0]

            for angle in angles:
                boxes_at_angle = self._detect_at_angle(image_path, angle)
                all_boxes.extend(boxes_at_angle)

            # Apply Non-Maximum Suppression to remove duplicate detections
            if len(all_boxes) > 0:
                boxes = self._nms_boxes(all_boxes, iou_threshold=0.5)
            else:
                boxes = []

            detection_method = "multi-angle detection" if use_multiangle else "single-angle detection"
            logger.info(f"Detected {len(boxes)} logo(s) after {detection_method}")
            return boxes

        except Exception as e:
            logger.error(f"Error during logo detection: {e}")
            return []
