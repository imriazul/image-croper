"""Application configuration and constants."""

from dataclasses import dataclass, field
from typing import Tuple
import os


@dataclass
class PassportConfig:
    """Configuration for passport photo processing."""

    # Passport dimensions
    width_mm: float = 45.0
    height_mm: float = 55.0
    dpi: int = 300

    # Calculated pixel dimensions
    @property
    def width_px(self) -> int:
        return int(self.width_mm / 25.4 * self.dpi)

    @property
    def height_px(self) -> int:
        return int(self.height_mm / 25.4 * self.dpi)

    # Background color (pure white)
    background_color: Tuple[int, int, int] = (255, 255, 255)

    # Output format
    output_format: str = "JPG"
    jpeg_quality: int = 98

    # Face detection thresholds
    min_face_confidence: float = 0.5
    min_face_size_ratio: float = 0.015  # Min face area relative to image
    max_faces_allowed: int = 5  # Max faces before ambiguity rejection

    # Head positioning (ratios relative to output image)
    head_top_margin_ratio: float = 0.075  # Space above hair top (~7.5% of output height)
    head_height_ratio_min: float = 0.55  # Min head height ratio
    head_height_ratio_max: float = 0.75  # Max head height ratio
    head_center_tolerance: float = 0.10  # Horizontal center tolerance

    # Quality validation
    min_resolution: Tuple[int, int] = (150, 200)  # Min input resolution
    min_eye_visibility_confidence: float = 0.3
    min_face_area_px: int = 2500  # Min face area in pixels

    # Processing
    max_concurrent_workers: int = 4
    output_subfolder_cropped: str = "Cropped"
    output_subfolder_rejected: str = "Rejected"
    output_folder_name: str = "Passport_Photo_Output"

    # Supported formats
    supported_formats: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")

    def get_output_path(self, base_path: str) -> str:
        return os.path.join(base_path, self.output_folder_name)

    def get_cropped_path(self, base_path: str) -> str:
        return os.path.join(self.get_output_path(base_path), self.output_subfolder_cropped)

    def get_rejected_path(self, base_path: str) -> str:
        return os.path.join(self.get_output_path(base_path), self.output_subfolder_rejected)


# Default configuration instance
DEFAULT_CONFIG = PassportConfig()
