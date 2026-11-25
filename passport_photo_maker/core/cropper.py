"""Intelligent passport photo cropping algorithm."""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from .face_detector import FaceLandmarks
from ..utils.config import PassportConfig

logger = logging.getLogger(__name__)


class PassportCropper:
    """
    Intelligent facial-landmark-based passport photo cropping.

    Dynamically calculates crop coordinates for each photo based on
    detected face landmarks to produce properly framed passport photos.
    """

    def __init__(self, config: Optional[PassportConfig] = None):
        self.config = config or PassportConfig()

    def crop(
        self,
        image: np.ndarray,
        face: FaceLandmarks,
        output_width: Optional[int] = None,
        output_height: Optional[int] = None,
        pad_value: int = 255,
        alpha_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Crop image to passport photo dimensions based on face landmarks.

        Args:
            image: BGR image as numpy array
            face: Detected face landmarks
            output_width: Output width in pixels (default from config)
            output_height: Output height in pixels (default from config)
            pad_value: Padding value when crop extends past image bounds
            alpha_mask: Optional foreground alpha mask [0, 1] to locate exact hair top

        Returns:
            Cropped image or None on failure
        """
        out_w = output_width or self.config.width_px
        out_h = output_height or self.config.height_px
        h, w = image.shape[:2]

        # Calculate the crop region
        crop_rect = self._calculate_crop_region(
            face, w, h, out_w, out_h, alpha_mask=alpha_mask
        )
        if crop_rect is None:
            logger.error("Failed to calculate crop region")
            return None

        x, y, cw, ch = crop_rect

        # Extract the crop region with padding if needed
        cropped = self._extract_with_padding(image, x, y, cw, ch, pad_value=pad_value)
        if cropped is None:
            return None

        # Resize to exact output dimensions using high-quality interpolation
        if cropped.shape[1] != out_w or cropped.shape[0] != out_h:
            # Choose interpolation method based on scale direction
            if cw * ch > out_w * out_h:
                interpolation = cv2.INTER_AREA  # Downscaling
            else:
                interpolation = cv2.INTER_LANCZOS4  # Upscaling
            cropped = cv2.resize(cropped, (out_w, out_h), interpolation=interpolation)

        return cropped

    def _find_hair_top(
        self,
        face: FaceLandmarks,
        alpha_mask: Optional[np.ndarray],
        face_height: float,
        img_w: int,
    ) -> float:
        """Find actual top of hair using alpha mask, falling back to landmark estimate."""
        fallback_top = face.forehead[1] - face_height * 0.28
        if alpha_mask is None:
            return fallback_top

        # Search horizontally across the head width above the forehead
        x1 = max(0, int(face.head_left))
        x2 = min(img_w, int(face.head_right))
        y_forehead = max(0, int(face.forehead[1]))
        y_search_top = max(0, int(face.forehead[1] - face_height * 0.65))

        if x2 <= x1 or y_forehead <= y_search_top:
            return fallback_top

        band = alpha_mask[y_search_top:y_forehead, x1:x2]
        # Require at least 10% of the head width to be foreground in a row to ignore stray noise
        min_fg_cols = max(3, int((x2 - x1) * 0.10))
        row_counts = np.sum(band > 0.5, axis=1)
        fg_rows = np.where(row_counts >= min_fg_cols)[0]

        if len(fg_rows) > 0:
            hair_top = float(y_search_top + fg_rows[0])
            # Sanity check: hair top shouldn't be below forehead
            return min(hair_top, face.forehead[1])

        return fallback_top

    def _calculate_crop_region(
        self,
        face: FaceLandmarks,
        img_w: int,
        img_h: int,
        out_w: int,
        out_h: int,
        alpha_mask: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Calculate the optimal crop region for passport photo.

        Returns (x, y, width, height) of the crop rectangle, or None on failure.
        """
        # Target aspect ratio
        aspect_ratio = out_w / out_h

        # Face height from chin to forehead (landmark-based)
        face_height = face.chin[1] - face.forehead[1]
        if face_height <= 0:
            face_height = face.head_height

        # Use actual hair top from segmentation mask when available
        estimated_head_top = self._find_hair_top(face, alpha_mask, face_height, img_w)
        estimated_head_bottom = face.chin[1]

        # Total head height (hair top to chin)
        total_head_height = max(estimated_head_bottom - estimated_head_top, face_height * 1.1)

        # Standard passport framing: full head occupies ~70% of output height,
        # leaving ~7.5% white margin above hair and ~22.5% for neck/shoulders
        target_head_ratio = 0.70
        crop_height = total_head_height / target_head_ratio

        # Ensure crop height is at least 2.0x face height for balanced shoulders
        min_crop_height = face_height * 2.0
        crop_height = max(crop_height, min_crop_height)

        # Calculate crop width from aspect ratio
        crop_width = crop_height * aspect_ratio

        # Center horizontally on the eyes
        face_center_x = (face.left_eye[0] + face.right_eye[0]) / 2.0
        crop_x = face_center_x - crop_width / 2.0

        # Position vertically with exact configured top margin above hair top
        top_margin = crop_height * self.config.head_top_margin_ratio
        crop_y = estimated_head_top - top_margin

        # Adjust for image boundaries
        crop_x, crop_y, crop_width, crop_height = self._adjust_to_bounds(
            crop_x, crop_y, crop_width, crop_height, img_w, img_h, face_center_x
        )

        # Validate the crop
        if crop_width < 50 or crop_height < 50:
            logger.error("Calculated crop region too small")
            return None

        return (
            int(round(crop_x)),
            int(round(crop_y)),
            int(round(crop_width)),
            int(round(crop_height)),
        )

    def _adjust_to_bounds(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        img_w: int,
        img_h: int,
        face_center_x: float,
    ) -> Tuple[float, float, float, float]:
        """Adjust crop region to fit within image boundaries, allowing top white padding."""
        aspect = w / h

        # Allow negative y (top headroom padding) when the original photo is
        # tightly cropped above the hair, so hair is never cut off or touching
        # the top edge.
        max_top_pad = h * 0.25
        if y < -max_top_pad:
            y = -max_top_pad

        # Keep horizontal crop inside image if possible, or allow slight padding
        max_side_pad = w * 0.15
        if x < -max_side_pad:
            x = -max_side_pad
        if x + w > img_w + max_side_pad:
            x = img_w + max_side_pad - w

        # If crop is wider than image + allowed padding, shrink to fit
        if w > img_w + 2 * max_side_pad:
            w = img_w + 2 * max_side_pad
            h = w / aspect
            x = face_center_x - w / 2.0

        # Keep bottom within image without pushing the head down and creating
        # excess empty space above the hair.
        if y + h > img_h:
            # Prefer shrinking height from the bottom to keep exact top headroom
            new_h = img_h - y
            if new_h >= h * 0.78:
                h = new_h
                w = h * aspect
                x = face_center_x - w / 2.0
            else:
                # Only shift up minimally if image is very short vertically
                y = img_h - h
                if y < -max_top_pad:
                    y = -max_top_pad
                    h = img_h - y
                    w = h * aspect
                    x = face_center_x - w / 2.0

        return x, y, w, h

    def _extract_with_padding(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        pad_value: int = 255,
    ) -> Optional[np.ndarray]:
        """
        Extract crop region, padding with pad_value if region extends beyond image.
        """
        img_h, img_w = image.shape[:2]

        # Calculate the overlap between crop region and image
        src_x1 = max(0, x)
        src_y1 = max(0, y)
        src_x2 = min(img_w, x + w)
        src_y2 = min(img_h, y + h)

        if src_x2 <= src_x1 or src_y2 <= src_y1:
            logger.error("No overlap between crop region and image")
            return None

        # Create output canvas
        canvas = np.full((h, w, 3), pad_value, dtype=np.uint8)

        # Calculate destination position in canvas
        dst_x1 = src_x1 - x
        dst_y1 = src_y1 - y
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        # Copy the image region
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

        return canvas

    def validate_crop(
        self,
        cropped_image: np.ndarray,
        face: FaceLandmarks,
        original_shape: Tuple[int, int],
    ) -> Tuple[bool, str]:
        """
        Validate that a cropped image meets passport photo requirements.

        Returns (is_valid, reason).
        """
        h, w = cropped_image.shape[:2]
        out_w = self.config.width_px
        out_h = self.config.height_px

        # Check dimensions
        if abs(w - out_w) > 2 or abs(h - out_h) > 2:
            return False, f"Wrong dimensions: {w}x{h}, expected {out_w}x{out_h}"

        # Check if image is not mostly blank
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        non_white = np.sum(gray < 240)
        total = gray.size
        content_ratio = non_white / total

        if content_ratio < 0.1:
            return False, "Cropped image appears mostly blank"

        if content_ratio > 0.95:
            return False, "Cropped image has insufficient white background"

        return True, ""
