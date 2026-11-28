"""Background removal module using rembg."""

import logging
import threading
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Try to import rembg
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logger.warning("rembg not available. Background removal will not work.")


class BackgroundRemover:
    """High-quality background removal and replacement with pure white."""

    def __init__(self, model_name: str = "u2net_human_seg"):
        """
        Initialize background remover.

        Args:
            model_name: rembg model name. Options:
                - 'u2net': General purpose (default)
                - 'u2net_human_seg': Optimized for humans
                - 'isnet-general-use': ISNet general
                - 'silueta': Lightweight
        """
        self.model_name = model_name
        self._session = None
        self._initialized = False
        self._lock = threading.Lock()
        self.last_error: str = ""

    def initialize(self):
        """Initialize the rembg session. Downloads model on first use."""
        if self._initialized:
            return

        if not REMBG_AVAILABLE:
            raise RuntimeError(
                "rembg is not installed. Please install it with: "
                "pip install rembg"
            )

        try:
            self._session = new_session(self.model_name)
            self._initialized = True
            logger.info(f"Background remover initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize rembg session: {e}")
            raise RuntimeError(f"Background removal model initialization failed: {e}")

    def remove_background(
        self,
        image: np.ndarray,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        refine_edges: bool = True,
        post_process_mask: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Remove background and replace with solid color.

        Returns BGR image with background replaced, or None on failure.
        """
        result = self.remove_background_with_mask(
            image,
            bg_color=bg_color,
            refine_edges=refine_edges,
            post_process_mask=post_process_mask,
        )
        if result is None:
            return None
        return result[0]

    def remove_background_with_mask(
        self,
        image: np.ndarray,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        refine_edges: bool = True,
        post_process_mask: bool = False,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Remove background and return (composited_bgr, alpha_mask_float32).
        """
        if not self._initialized:
            self.initialize()

        self.last_error = ""
        try:
            h, w = image.shape[:2]

            # Downscale large images for fast, consistent segmentation inference
            max_dim = 1024
            scale = min(1.0, max_dim / max(h, w))
            if scale < 1.0:
                infer_img = cv2.resize(
                    image,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                infer_img = image

            rgb_image = cv2.cvtColor(infer_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            with self._lock:
                result_pil = remove(
                    pil_image,
                    session=self._session,
                    post_process_mask=post_process_mask,
                )

            result_rgba = np.array(result_pil)
            if result_rgba.ndim != 3 or result_rgba.shape[2] != 4:
                self.last_error = "Segmentation did not return an RGBA mask"
                logger.error(self.last_error)
                return None

            alpha = result_rgba[:, :, 3].astype(np.float32) / 255.0

            # Resize alpha back to original image dimensions if scaled
            if scale < 1.0:
                alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)

            # Fill enclosed holes inside the foreground (e.g. white shirt panels)
            alpha = self._fill_interior_holes(alpha)

            # Refine edges with OpenCV
            if refine_edges:
                alpha = self._refine_edges(alpha)

            # Validate segmentation quality
            valid, reason = self._validate_segmentation(alpha)
            if not valid:
                self.last_error = reason
                logger.warning(f"Background segmentation rejected: {reason}")
                return None

            # Composite onto white background
            result = self._composite_on_background(image, alpha, bg_color)
            return result, alpha

        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Background removal failed: {e}", exc_info=True)
            return None

    def _fill_interior_holes(self, alpha: np.ndarray) -> np.ndarray:
        """Fill enclosed holes inside the foreground mask (e.g. white clothing panels)."""
        mask = (alpha > 0.5).astype(np.uint8)
        h, w = mask.shape
        ff = mask.copy()
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        for seed in ((0, 0), (w - 1, 0)):
            if ff[seed[1], seed[0]] == 0:
                cv2.floodFill(ff, flood_mask, seed, 2)
        holes = ff == 0
        if np.any(holes):
            alpha = np.where(holes, 1.0, alpha).astype(np.float32)
        return alpha

    def _refine_edges(self, alpha: np.ndarray) -> np.ndarray:
        """Refine alpha mask edges for smooth, natural hair and clothing cutouts."""
        # Slightly tighten low-alpha fringe to eliminate background halos
        alpha = np.clip((alpha - 0.06) / 0.90, 0.0, 1.0).astype(np.float32)

        alpha_uint8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)

        # Clean isolated 1px speckles in background without blunting hair edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(alpha_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
        # Keep original soft alpha along boundaries, only remove isolated specks
        alpha_uint8 = np.where(opened == 0, 0, alpha_uint8).astype(np.uint8)

        # Gentle Gaussian feathering for anti-aliased, non-jagged hair edges
        feathered = cv2.GaussianBlur(alpha_uint8, (3, 3), 0.75)

        return feathered.astype(np.float32) / 255.0

    def _validate_segmentation(self, alpha: np.ndarray) -> Tuple[bool, str]:
        """Validate that segmentation produced a reasonable result."""
        h, w = alpha.shape
        total_pixels = h * w

        fg_pixels = np.sum(alpha > 0.5)
        fg_ratio = fg_pixels / total_pixels

        if fg_ratio < 0.02:
            return False, f"Foreground too small ({fg_ratio:.1%})"

        if fg_ratio > 0.99:
            return False, f"Foreground too large ({fg_ratio:.1%})"

        fg_mask = (alpha > 0.5).astype(np.uint8)
        moments = cv2.moments(fg_mask)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            center_x_ratio = cx / w
            if center_x_ratio < 0.15 or center_x_ratio > 0.85:
                return False, f"Foreground not centered (x={center_x_ratio:.2f})"

        return True, ""

    def _composite_on_background(
        self,
        image: np.ndarray,
        alpha: np.ndarray,
        bg_color: Tuple[int, int, int],
    ) -> np.ndarray:
        """Composite foreground onto solid background color."""
        h, w = image.shape[:2]
        background = np.full((h, w, 3), bg_color[::-1], dtype=np.uint8)
        alpha_3ch = np.stack([alpha] * 3, axis=-1)
        result = (
            image.astype(np.float32) * alpha_3ch
            + background.astype(np.float32) * (1.0 - alpha_3ch)
        )
        return np.clip(result, 0, 255).astype(np.uint8)

    def get_alpha_mask(
        self, image: np.ndarray, post_process_mask: bool = False
    ) -> Optional[np.ndarray]:
        """Get just the alpha mask without compositing."""
        res = self.remove_background_with_mask(
            image, refine_edges=False, post_process_mask=post_process_mask
        )
        return res[1] if res is not None else None

    def close(self):
        """Release resources."""
        self._session = None
        self._initialized = False
