"""Image enhancement and quality validation module."""

import logging
from typing import Optional, Tuple, List

import cv2
import numpy as np

from .face_detector import FaceLandmarks
from ..utils.config import PassportConfig

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """
    Non-destructive image enhancement for passport photos.

    Applies gentle, conditional adjustments only when needed so clean,
    well-exposed photos retain their full original sharpness, skin texture,
    and natural colors.
    """

    def enhance(
        self,
        image: np.ndarray,
        auto_brightness: bool = True,
        auto_contrast: bool = True,
        auto_white_balance: bool = False,
        noise_reduction: bool = False,
        sharpen: bool = True,
    ) -> np.ndarray:
        """
        Apply gentle, non-destructive enhancements to passport photo.
        """
        result = image.copy()

        if auto_white_balance:
            result = self._auto_white_balance(result)

        if auto_brightness:
            result = self._auto_brightness(result)

        if auto_contrast:
            result = self._auto_contrast(result)

        if noise_reduction:
            result = self._reduce_noise(result)

        if sharpen:
            result = self._sharpen(result)

        return result

    def _auto_white_balance(self, image: np.ndarray) -> np.ndarray:
        """Apply very subtle white balance correction only for strong casts."""
        result = image.astype(np.float32)
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        avg_all = (avg_b + avg_g + avg_r) / 3.0

        if avg_b > 0 and avg_g > 0 and avg_r > 0:
            # Clamp channel scaling tightly so colored clothing doesn't tint skin/shirts
            scale_b = np.clip(avg_all / avg_b, 0.94, 1.06)
            scale_g = np.clip(avg_all / avg_g, 0.94, 1.06)
            scale_r = np.clip(avg_all / avg_r, 0.94, 1.06)
            result[:, :, 0] *= scale_b
            result[:, :, 1] *= scale_g
            result[:, :, 2] *= scale_r

        return np.clip(result, 0, 255).astype(np.uint8)

    def _auto_brightness(self, image: np.ndarray) -> np.ndarray:
        """Only gently brighten underexposed photos; never darken well-lit photos."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.float32)
        mean_v = np.mean(v)

        # Only brighten if noticeably dark (underexposed)
        if mean_v >= 105.0:
            return image

        target = 115.0
        ratio = np.clip(target / max(mean_v, 1.0), 1.0, 1.25)
        v = np.clip(v * ratio, 0, 255).astype(np.uint8)
        hsv[:, :, 2] = v

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _auto_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply very subtle contrast enhancement blended with original."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        # Blend 25% CLAHE + 75% original to preserve natural tones
        l_out = cv2.addWeighted(l, 0.75, l_clahe, 0.25, 0)

        lab = cv2.merge([l_out, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply very light noise reduction."""
        return cv2.fastNlMeansDenoisingColored(image, None, 2, 2, 7, 21)

    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """Apply luminance-only unsharp mask to sharpen details without color halos."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        blurred_l = cv2.GaussianBlur(l, (0, 0), 0.85)
        sharpened_l = cv2.addWeighted(l, 1.24, blurred_l, -0.24, 0)

        lab = cv2.merge([sharpened_l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class QualityValidator:
    """
    Validates processed passport photos against quality requirements.

    Checks face positioning, dimensions, background, and overall quality.
    """

    def __init__(self, config: Optional[PassportConfig] = None):
        self.config = config or PassportConfig()

    def validate(
        self,
        image: np.ndarray,
        face: Optional[FaceLandmarks] = None,
        check_background: bool = True,
    ) -> Tuple[bool, List[str]]:
        """
        Validate a processed passport photo.

        Args:
            image: Processed BGR image
            face: Face landmarks (if available from re-detection)
            check_background: Whether to validate background whiteness

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        h, w = image.shape[:2]

        # 1. Check dimensions
        expected_w = self.config.width_px
        expected_h = self.config.height_px
        if abs(w - expected_w) > 2 or abs(h - expected_h) > 2:
            issues.append(f"Wrong dimensions: {w}x{h}, expected {expected_w}x{expected_h}")

        # 2. Check resolution
        if w < self.config.min_resolution[0] or h < self.config.min_resolution[1]:
            issues.append(f"Resolution too low: {w}x{h}")

        # 3. Check if image has content (not blank)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        non_white = np.sum(gray < 240)
        content_ratio = non_white / gray.size
        if content_ratio < 0.08:
            issues.append("Image appears mostly blank")

        # 4. Check background whiteness
        if check_background:
            bg_issues = self._check_background(image)
            issues.extend(bg_issues)

        # 5. Check image is not corrupted / all black
        if np.mean(gray) < 10:
            issues.append("Image appears to be all black / corrupted")

        # 6. Check for extreme color cast
        b, g, r = cv2.split(image)
        mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)
        channel_diff = max(abs(mean_b - mean_g), abs(mean_g - mean_r), abs(mean_b - mean_r))
        if channel_diff > 50:
            issues.append("Strong color cast detected")

        is_valid = len(issues) == 0
        return is_valid, issues

    def _check_background(self, image: np.ndarray) -> List[str]:
        """Check that background corners and upper edges are pure white."""
        issues = []
        h, w = image.shape[:2]

        # Sample top-left and top-right corners and upper side edges
        # (avoid top-center where tall hair can occasionally reach near top)
        margin = max(5, int(min(w, h) * 0.03))
        regions = [
            image[0:margin, 0:margin],                      # Top-left corner
            image[0:margin, w - margin:w],                  # Top-right corner
            image[margin:margin * 3, 0:margin],             # Upper-left edge
            image[margin:margin * 3, w - margin:w],         # Upper-right edge
        ]

        for i, region in enumerate(regions):
            if region.size == 0:
                continue
            mean_val = np.mean(region)
            if mean_val < 240:
                issues.append(f"Background not pure white in region {i} (mean={mean_val:.0f})")
                break

        return issues

    def validate_export(self, filepath: str) -> Tuple[bool, str]:
        """Validate that an exported file is readable and correct."""
        try:
            img = cv2.imread(filepath)
            if img is None:
                return False, "Exported file cannot be read"

            h, w = img.shape[:2]
            if w == 0 or h == 0:
                return False, "Exported file has zero dimensions"

            return True, ""
        except Exception as e:
            return False, f"Export validation error: {e}"
