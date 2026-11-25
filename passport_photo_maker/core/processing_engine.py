"""Bulk processing engine with multithreaded execution."""

import csv
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Event, Lock
from typing import Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from .face_detector import FaceDetector, FaceDetectionResult
from .background_remover import BackgroundRemover
from .cropper import PassportCropper
from .enhancer import ImageEnhancer, QualityValidator
from .input_handler import InputHandler, InputScanResult
from ..utils.config import PassportConfig

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ImageResult:
    """Result of processing a single image."""

    filename: str = ""
    source_path: str = ""
    output_path: str = ""
    success: bool = False
    rejection_reason: str = ""
    processing_time: float = 0.0
    error: str = ""


@dataclass
class BatchProgress:
    """Live batch processing progress."""

    total: int = 0
    processed: int = 0
    successful: int = 0
    rejected: int = 0
    errors: int = 0
    remaining: int = 0
    current_file: str = ""
    percentage: float = 0.0
    elapsed_time: float = 0.0
    status: ProcessingStatus = ProcessingStatus.IDLE


@dataclass
class BatchReport:
    """Final batch processing report."""

    total_found: int = 0
    successfully_cropped: int = 0
    rejected: int = 0
    processing_errors: int = 0
    total_time: float = 0.0
    results: List[ImageResult] = field(default_factory=list)
    output_folder: str = ""
    cropped_folder: str = ""
    rejected_folder: str = ""


class ProcessingEngine:
    """
    Bulk passport photo processing engine.

    Orchestrates face detection, background removal, cropping,
    enhancement, and validation for batches of photographs.
    """

    def __init__(self, config: Optional[PassportConfig] = None):
        self.config = config or PassportConfig()

        # Processing modules
        self._face_detector: Optional[FaceDetector] = None
        self._bg_remover: Optional[BackgroundRemover] = None
        self._cropper: Optional[PassportCropper] = None
        self._enhancer: Optional[ImageEnhancer] = None
        self._validator: Optional[QualityValidator] = None
        self._input_handler: Optional[InputHandler] = None

        # State
        self._status = ProcessingStatus.IDLE
        self._cancel_event = Event()
        self._pause_event = Event()
        self._pause_event.set()  # Not paused initially
        self._progress = BatchProgress()
        self._progress_lock = Lock()
        self._used_output_names: Set[str] = set()

        # Callbacks
        self._on_progress: Optional[Callable[[BatchProgress], None]] = None
        self._on_image_complete: Optional[Callable[[ImageResult], None]] = None
        self._on_batch_complete: Optional[Callable[[BatchReport], None]] = None

    def set_callbacks(
        self,
        on_progress: Optional[Callable[[BatchProgress], None]] = None,
        on_image_complete: Optional[Callable[[ImageResult], None]] = None,
        on_batch_complete: Optional[Callable[[BatchReport], None]] = None,
    ):
        """Set callback functions for progress updates."""
        self._on_progress = on_progress
        self._on_image_complete = on_image_complete
        self._on_batch_complete = on_batch_complete

    def initialize(self):
        """Initialize all processing modules."""
        logger.info("Initializing processing engine...")

        self._face_detector = FaceDetector(
            min_confidence=self.config.min_face_confidence,
            max_faces=self.config.max_faces_allowed,
        )
        self._face_detector.initialize()

        self._bg_remover = BackgroundRemover(model_name="u2net")
        self._bg_remover.initialize()

        self._cropper = PassportCropper(self.config)
        self._enhancer = ImageEnhancer()
        self._validator = QualityValidator(self.config)
        self._input_handler = InputHandler()

        logger.info("Processing engine initialized successfully")

    def process_batch(
        self,
        scan_result: InputScanResult,
        output_base_path: str,
        max_workers: Optional[int] = None,
    ) -> BatchReport:
        """
        Process a batch of photographs.

        Args:
            scan_result: Scanned input with image paths
            output_base_path: Base path for output folders
            max_workers: Max concurrent workers (default from config)

        Returns:
            BatchReport with results
        """
        if self._status == ProcessingStatus.RUNNING:
            raise RuntimeError("Processing already in progress")

        workers = max_workers or self.config.max_concurrent_workers
        self._status = ProcessingStatus.RUNNING
        self._cancel_event.clear()
        self._pause_event.set()
        self._used_output_names.clear()

        # Create output directories
        output_dir = self.config.get_output_path(output_base_path)
        cropped_dir = self.config.get_cropped_path(output_base_path)
        rejected_dir = self.config.get_rejected_path(output_base_path)
        os.makedirs(cropped_dir, exist_ok=True)
        os.makedirs(rejected_dir, exist_ok=True)

        # Initialize progress
        total = len(scan_result.image_paths)
        with self._progress_lock:
            self._progress = BatchProgress(
                total=total,
                remaining=total,
                status=ProcessingStatus.RUNNING,
            )

        report = BatchReport(
            total_found=total,
            output_folder=output_dir,
            cropped_folder=cropped_dir,
            rejected_folder=rejected_dir,
        )

        start_time = time.time()
        results: List[ImageResult] = []

        try:
            # Process images with controlled concurrency
            # Use ThreadPoolExecutor for I/O-bound work with controlled thread count
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures: Dict[Future, str] = {}

                for image_path in scan_result.image_paths:
                    if self._cancel_event.is_set():
                        break

                    future = executor.submit(
                        self._process_single_image,
                        image_path,
                        cropped_dir,
                        rejected_dir,
                    )
                    futures[future] = image_path

                # Collect results
                for future in futures:
                    if self._cancel_event.is_set():
                        future.cancel()
                        continue

                    # Wait for pause to be released
                    self._pause_event.wait()

                    try:
                        result = future.result(timeout=120)  # 2 min timeout per image
                        results.append(result)

                        with self._progress_lock:
                            self._progress.processed += 1
                            if result.success:
                                self._progress.successful += 1
                            elif result.error:
                                self._progress.errors += 1
                            else:
                                self._progress.rejected += 1
                            self._progress.remaining = total - self._progress.processed
                            self._progress.current_file = result.filename
                            self._progress.percentage = (
                                self._progress.processed / total * 100
                                if total > 0
                                else 0
                            )
                            self._progress.elapsed_time = time.time() - start_time

                        if self._on_image_complete:
                            self._on_image_complete(result)
                        if self._on_progress:
                            self._on_progress(self._progress)

                    except Exception as e:
                        logger.error(f"Future error for {futures[future]}: {e}")
                        error_result = ImageResult(
                            filename=os.path.basename(futures[future]),
                            source_path=futures[future],
                            error=str(e),
                        )
                        results.append(error_result)

                        with self._progress_lock:
                            self._progress.processed += 1
                            self._progress.errors += 1
                            self._progress.remaining = total - self._progress.processed
                            self._progress.percentage = (
                                self._progress.processed / total * 100
                                if total > 0
                                else 0
                            )

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            self._status = ProcessingStatus.ERROR

        # Build report
        total_time = time.time() - start_time
        report.total_time = total_time
        report.results = results
        report.successfully_cropped = sum(1 for r in results if r.success)
        report.rejected = sum(1 for r in results if not r.success and not r.error)
        report.processing_errors = sum(1 for r in results if r.error)

        if not self._cancel_event.is_set():
            self._status = ProcessingStatus.COMPLETED
        else:
            self._status = ProcessingStatus.IDLE

        with self._progress_lock:
            self._progress.status = self._status
            self._progress.percentage = 100.0 if not self._cancel_event.is_set() else self._progress.percentage

        if self._on_batch_complete:
            self._on_batch_complete(report)

        return report

    def _process_single_image(
        self,
        image_path: str,
        cropped_dir: str,
        rejected_dir: str,
    ) -> ImageResult:
        """Process a single photograph through the full pipeline."""
        filename = os.path.basename(image_path)
        result = ImageResult(filename=filename, source_path=image_path)
        start_time = time.time()

        try:
            # 1. Load image
            image = cv2.imread(image_path)
            if image is None:
                result.rejection_reason = "Failed to load image (corrupted or unsupported)"
                self._save_rejected(image_path, rejected_dir, result.rejection_reason)
                result.processing_time = time.time() - start_time
                return result

            h, w = image.shape[:2]

            # Check minimum resolution
            if w < self.config.min_resolution[0] or h < self.config.min_resolution[1]:
                result.rejection_reason = f"Resolution too low ({w}x{h})"
                self._save_rejected(image_path, rejected_dir, result.rejection_reason)
                result.processing_time = time.time() - start_time
                return result

            # 2. Detect face
            detection = self._face_detector.detect(
                image,
                min_face_size_ratio=self.config.min_face_size_ratio,
                min_face_area_px=self.config.min_face_area_px,
            )

            if not detection.success:
                result.rejection_reason = detection.rejection_reason
                self._save_rejected(image_path, rejected_dir, result.rejection_reason)
                result.processing_time = time.time() - start_time
                return result

            face = detection.primary_face

            # 3. Remove background (returns composited BGR and alpha mask)
            bg_res = self._bg_remover.remove_background_with_mask(
                image,
                bg_color=self.config.background_color,
                refine_edges=True,
            )

            if bg_res is None:
                reason = self._bg_remover.last_error or "Background removal failed"
                result.rejection_reason = f"Background removal failed: {reason}"
                self._save_rejected(image_path, rejected_dir, result.rejection_reason)
                result.processing_time = time.time() - start_time
                return result

            _, alpha = bg_res

            # 4. Crop original image and alpha mask to passport dimensions
            #    using the alpha mask to locate the exact top of the hair
            cropped_fg = self._cropper.crop(image, face, alpha_mask=alpha)
            if cropped_fg is None:
                result.rejection_reason = "Cropping failed - could not calculate crop region"
                self._save_rejected(image_path, rejected_dir, result.rejection_reason)
                result.processing_time = time.time() - start_time
                return result

            # Crop alpha mask using the same cropper geometry
            alpha_3ch = np.stack([alpha] * 3, axis=-1)
            alpha_uint8 = np.clip(alpha_3ch * 255.0, 0, 255).astype(np.uint8)
            cropped_alpha_3ch = self._cropper.crop(
                alpha_uint8, face, pad_value=0, alpha_mask=alpha
            )
            if cropped_alpha_3ch is None:
                result.rejection_reason = "Cropping alpha mask failed"
                self._save_rejected(image_path, rejected_dir, result.rejection_reason)
                result.processing_time = time.time() - start_time
                return result
            cropped_alpha = cropped_alpha_3ch[:, :, 0].astype(np.float32) / 255.0

            # 5. Enhance foreground subject BEFORE compositing onto pure white
            #    so the background stays exact RGB(255, 255, 255)
            enhanced_fg = self._enhancer.enhance(cropped_fg)
            enhanced = self._bg_remover._composite_on_background(
                enhanced_fg,
                cropped_alpha,
                self.config.background_color,
            )

            # 6. Validate quality
            is_valid, issues = self._validator.validate(enhanced, face)

            if not is_valid:
                result.rejection_reason = "; ".join(issues)
                self._save_rejected(image_path, rejected_dir, result.rejection_reason)
                result.processing_time = time.time() - start_time
                return result

            # 7. Save to cropped folder
            output_path = self._input_handler.get_unique_output_name(
                image_path,
                cropped_dir,
                self._used_output_names,
                output_ext=f".{self.config.output_format.lower()}",
            )

            # Save with appropriate quality
            if self.config.output_format.upper() in ("JPG", "JPEG"):
                cv2.imwrite(
                    output_path,
                    enhanced,
                    [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
                )
            elif self.config.output_format.upper() == "PNG":
                cv2.imwrite(output_path, enhanced)
            else:
                cv2.imwrite(output_path, enhanced)

            # 8. Validate export
            export_valid, export_error = self._validator.validate_export(output_path)
            if not export_valid:
                result.rejection_reason = f"Export validation failed: {export_error}"
                # Clean up bad export
                if os.path.exists(output_path):
                    os.remove(output_path)
                self._save_rejected(image_path, rejected_dir, result.rejection_reason)
                result.processing_time = time.time() - start_time
                return result

            result.success = True
            result.output_path = output_path
            result.processing_time = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}", exc_info=True)
            result.error = str(e)
            try:
                self._save_rejected(image_path, rejected_dir, f"Processing error: {e}")
            except Exception:
                pass
            result.processing_time = time.time() - start_time
            return result

    def _save_rejected(
        self, source_path: str, rejected_dir: str, reason: str
    ):
        """Save rejected image to rejected folder with reason."""
        try:
            filename = os.path.basename(source_path)
            dest_path = os.path.join(rejected_dir, filename)

            # Handle filename collision
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(rejected_dir, f"{base}_{counter}{ext}")
                    counter += 1

            shutil.copy2(source_path, dest_path)

            # Save rejection reason to a text file
            reason_file = dest_path + ".reason.txt"
            with open(reason_file, "w") as f:
                f.write(f"File: {filename}\n")
                f.write(f"Reason: {reason}\n")

        except Exception as e:
            logger.error(f"Failed to save rejected file {source_path}: {e}")

    def cancel(self):
        """Cancel the current batch processing."""
        self._cancel_event.set()
        self._pause_event.set()  # Unpause if paused
        self._status = ProcessingStatus.CANCELLING
        logger.info("Processing cancellation requested")

    def pause(self):
        """Pause processing."""
        if self._status == ProcessingStatus.RUNNING:
            self._pause_event.clear()
            self._status = ProcessingStatus.PAUSED
            with self._progress_lock:
                self._progress.status = ProcessingStatus.PAUSED
            logger.info("Processing paused")

    def resume(self):
        """Resume processing."""
        if self._status == ProcessingStatus.PAUSED:
            self._pause_event.set()
            self._status = ProcessingStatus.RUNNING
            with self._progress_lock:
                self._progress.status = ProcessingStatus.RUNNING
            logger.info("Processing resumed")

    def get_progress(self) -> BatchProgress:
        """Get current processing progress (thread-safe)."""
        with self._progress_lock:
            return BatchProgress(
                total=self._progress.total,
                processed=self._progress.processed,
                successful=self._progress.successful,
                rejected=self._progress.rejected,
                errors=self._progress.errors,
                remaining=self._progress.remaining,
                current_file=self._progress.current_file,
                percentage=self._progress.percentage,
                elapsed_time=self._progress.elapsed_time,
                status=self._progress.status,
            )

    @property
    def status(self) -> ProcessingStatus:
        return self._status

    def generate_csv_report(self, report: BatchReport, output_path: str):
        """Generate a CSV processing report."""
        try:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Filename",
                    "Source Path",
                    "Status",
                    "Output Path",
                    "Rejection Reason",
                    "Error",
                    "Processing Time (s)",
                ])

                for result in report.results:
                    writer.writerow([
                        result.filename,
                        result.source_path,
                        "Success" if result.success else ("Error" if result.error else "Rejected"),
                        result.output_path or "",
                        result.rejection_reason or "",
                        result.error or "",
                        f"{result.processing_time:.2f}",
                    ])

                # Summary row
                writer.writerow([])
                writer.writerow(["SUMMARY"])
                writer.writerow(["Total Found", report.total_found])
                writer.writerow(["Successfully Cropped", report.successfully_cropped])
                writer.writerow(["Rejected", report.rejected])
                writer.writerow(["Processing Errors", report.processing_errors])
                writer.writerow(["Total Time (s)", f"{report.total_time:.2f}"])

            logger.info(f"CSV report saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate CSV report: {e}")

    def cleanup(self):
        """Release all resources."""
        if self._face_detector:
            self._face_detector.close()
        if self._bg_remover:
            self._bg_remover.close()
        if self._input_handler:
            self._input_handler.cleanup()
