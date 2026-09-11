"""
AquaVision AI — Desktop Application Entry Point
================================================
Main entry point to launch the PySide6 desktop GUI application for
Fish4Knowledge species classification.

Usage:
    python app.py
"""

import sys
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

# Ensure root directory is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.main_window import MainWindow, DEFAULT_DATA_ROOT


def main():
    # Enable High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("AquaVision AI")
    app.setOrganizationName("Fish4Knowledge")

    # Command line argument for data root if specified
    data_root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_ROOT

    window = MainWindow(data_root=data_root)
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
