import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt  # Add this import for Qt namespace
from PyQt5.QtGui import QFontDatabase

from ui.main_window import MainWindow

def ensure_resources():
    """Ensure the necessary resource directories exist"""
    os.makedirs("ui/resources", exist_ok=True)
    
    # Create basic font files if they don't exist (in a real app, you'd include these)
    # For this demo, we'll use system fonts if the specified ones don't exist
    pass

def main():
    # Create app instance
    app = QApplication(sys.argv)
    
    # Apply high DPI scaling
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    # Ensure resource directories exist
    ensure_resources()
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()