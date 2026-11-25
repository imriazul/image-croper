#!/usr/bin/env python3
"""AI Passport Photo Maker - Main Entry Point.

A professional AI-powered desktop application for automatically processing
bulk passport photographs with face detection, background removal, and
intelligent cropping.
"""

import logging
import sys
import os

# Configure logging before importing anything else
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PassportPhotoMaker")


def main():
    """Launch the AI Passport Photo Maker application."""
    # Suppress noisy library logs
    logging.getLogger("mediapipe").setLevel(logging.WARNING)
    logging.getLogger("rembg").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)

    logger.info("Starting AI Passport Photo Maker...")

    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QFont
    except ImportError:
        print(
            "ERROR: PySide6 is not installed.\n"
            "Please install dependencies:\n"
            "  pip install -r requirements.txt"
        )
        sys.exit(1)

    # Create application
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("AI Passport Photo Maker")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("PassportPhotoMaker")

    # Set default font
    font = QFont("Segoe UI", 10)
    if sys.platform == "darwin":
        font = QFont("SF Pro Display", 12)
    app.setFont(font)

    # Apply stylesheet
    from passport_photo_maker.ui.main_window import STYLESHEET
    app.setStyleSheet(STYLESHEET)

    # Create and show main window
    from passport_photo_maker.ui.main_window import MainWindow
    window = MainWindow()
    window.show()

    logger.info("Application window shown")

    # Run event loop
    exit_code = app.exec()

    # Cleanly tear down Qt objects before Python atexit runs PySide6 cleanup
    import gc
    del window
    gc.collect()
    app.processEvents()
    del app
    gc.collect()

    logger.info("Application exiting")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
