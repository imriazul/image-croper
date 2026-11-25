"""Input handling: folder scanning and ZIP extraction."""

import logging
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class InputScanResult:
    """Result of scanning input for photographs."""

    image_paths: List[str] = field(default_factory=list)
    total_found: int = 0
    duplicates: List[str] = field(default_factory=list)
    unsupported: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    source_type: str = ""  # "folder" or "zip"
    source_path: str = ""
    temp_dir: Optional[str] = None  # For ZIP extraction


class InputHandler:
    """Handles folder scanning and ZIP extraction for input photographs."""

    def __init__(self):
        self._temp_dirs: List[str] = []

    def scan_folder(self, folder_path: str) -> InputScanResult:
        """
        Recursively scan a folder for supported image files.

        Args:
            folder_path: Path to the folder to scan

        Returns:
            InputScanResult with found images
        """
        result = InputScanResult(source_type="folder", source_path=folder_path)

        if not os.path.isdir(folder_path):
            result.errors.append(f"Folder does not exist: {folder_path}")
            return result

        seen_names: Dict[str, str] = {}  # filename -> first path

        for root, dirs, files in os.walk(folder_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for filename in sorted(files):
                if filename.startswith("."):
                    continue

                filepath = os.path.join(root, filename)
                ext = os.path.splitext(filename)[1].lower()

                if ext not in SUPPORTED_EXTENSIONS:
                    result.unsupported.append(filepath)
                    continue

                # Check for duplicates
                base_name = filename.lower()
                if base_name in seen_names:
                    result.duplicates.append(filepath)
                    # Still include it but with a modified name
                    result.image_paths.append(filepath)
                else:
                    seen_names[base_name] = filepath
                    result.image_paths.append(filepath)

        result.total_found = len(result.image_paths)
        logger.info(
            f"Folder scan: {result.total_found} images found, "
            f"{len(result.duplicates)} duplicates, "
            f"{len(result.unsupported)} unsupported"
        )
        return result

    def scan_zip(self, zip_path: str) -> InputScanResult:
        """
        Extract and scan a ZIP archive for supported image files.

        Args:
            zip_path: Path to the ZIP file

        Returns:
            InputScanResult with found images
        """
        result = InputScanResult(source_type="zip", source_path=zip_path)

        if not os.path.isfile(zip_path):
            result.errors.append(f"ZIP file does not exist: {zip_path}")
            return result

        if not zipfile.is_zipfile(zip_path):
            result.errors.append(f"Not a valid ZIP file: {zip_path}")
            return result

        # Create temporary directory for extraction
        temp_dir = tempfile.mkdtemp(prefix="passport_photo_")
        self._temp_dirs.append(temp_dir)
        result.temp_dir = temp_dir

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Check for zip bomb
                total_size = sum(info.file_size for info in zf.infolist())
                if total_size > 10 * 1024 * 1024 * 1024:  # 10GB limit
                    result.errors.append("ZIP file is too large (>10GB)")
                    return result

                # Extract safely
                for info in zf.infolist():
                    # Skip directories
                    if info.is_dir():
                        continue

                    # Security: prevent path traversal
                    name = info.filename
                    if name.startswith("/") or ".." in name:
                        logger.warning(f"Skipping unsafe path: {name}")
                        continue

                    # Skip hidden/system files
                    basename = os.path.basename(name)
                    if basename.startswith(".") or basename.startswith("__"):
                        continue

                    ext = os.path.splitext(basename)[1].lower()
                    if ext not in SUPPORTED_EXTENSIONS:
                        result.unsupported.append(name)
                        continue

                    # Extract the file
                    try:
                        # Create flattened structure to avoid nested dirs
                        # Use relative path as prefix to avoid name collisions
                        safe_name = name.replace("/", "_").replace("\\", "_")
                        target_path = os.path.join(temp_dir, safe_name)

                        # Handle duplicates
                        if os.path.exists(target_path):
                            base, ext_part = os.path.splitext(safe_name)
                            counter = 1
                            while os.path.exists(target_path):
                                target_path = os.path.join(
                                    temp_dir, f"{base}_{counter}{ext_part}"
                                )
                                counter += 1
                            result.duplicates.append(name)

                        # Extract to target
                        with zf.open(info) as src, open(target_path, "wb") as dst:
                            shutil.copyfileobj(src, dst)

                        result.image_paths.append(target_path)

                    except Exception as e:
                        logger.error(f"Failed to extract {name}: {e}")
                        result.errors.append(f"Extract failed: {name} - {e}")

        except zipfile.BadZipFile:
            result.errors.append("Corrupted ZIP file")
        except Exception as e:
            result.errors.append(f"ZIP processing error: {e}")

        result.total_found = len(result.image_paths)
        logger.info(
            f"ZIP scan: {result.total_found} images extracted, "
            f"{len(result.duplicates)} duplicates, "
            f"{len(result.unsupported)} unsupported"
        )
        return result

    def get_unique_output_name(
        self,
        original_path: str,
        output_dir: str,
        existing_names: Set[str],
        output_ext: str = ".jpg",
    ) -> str:
        """
        Generate a unique output filename, preventing collisions.

        Args:
            original_path: Original file path
            output_dir: Output directory
            existing_names: Set of already-used names (lowercase)
            output_ext: Output file extension

        Returns:
            Full path with unique filename
        """
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        name = f"{base_name}{output_ext}"
        lower_name = name.lower()

        if lower_name not in existing_names:
            existing_names.add(lower_name)
            return os.path.join(output_dir, name)

        # Append counter for uniqueness
        counter = 1
        while True:
            name = f"{base_name}_{counter}{output_ext}"
            lower_name = name.lower()
            if lower_name not in existing_names:
                existing_names.add(lower_name)
                return os.path.join(output_dir, name)
            counter += 1

    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self._temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temp dir: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up {temp_dir}: {e}")
        self._temp_dirs.clear()
