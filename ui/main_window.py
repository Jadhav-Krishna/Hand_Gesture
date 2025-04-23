import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFrame, QGraphicsDropShadowEffect, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal, pyqtSlot, QPropertyAnimation, QRect, QEasingCurve
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont, QPalette, QFontDatabase, QIcon

from ui.camera_widget import CameraWidget
from ui.result_widget import ResultWidget
from utils.gesture_history import GestureHistory

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Gesture Recognition")
        self.setMinimumSize(1200, 800)
        
        # Set up modern theme
        self.setup_styles()
        
        # Initialize central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Create left panel (camera and controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(16)
        
        # Create camera widget
        self.camera_widget = CameraWidget()
        self.camera_widget.setMinimumWidth(640)
        self.camera_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Camera container with title
        camera_container = QFrame()
        camera_container.setObjectName("cameraContainer")
        camera_layout = QVBoxLayout(camera_container)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        
        # Camera header
        camera_header = QWidget()
        camera_header.setObjectName("panelHeader")
        camera_header_layout = QHBoxLayout(camera_header)
        camera_header_layout.setContentsMargins(16, 8, 16, 8)
        
        camera_title = QLabel("Camera Feed")
        camera_title.setObjectName("panelTitle")
        camera_header_layout.addWidget(camera_title)
        
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setObjectName("statsLabel")
        camera_header_layout.addWidget(self.fps_label, 0, Qt.AlignRight)
        
        camera_layout.addWidget(camera_header)
        camera_layout.addWidget(self.camera_widget)
        
        left_layout.addWidget(camera_container)
        
        # Controls panel
        controls_panel = QFrame()
        controls_panel.setObjectName("controlsPanel")
        controls_layout = QVBoxLayout(controls_panel)
        
        # Controls header
        controls_header = QWidget()
        controls_header.setObjectName("panelHeader")
        controls_header_layout = QHBoxLayout(controls_header)
        controls_header_layout.setContentsMargins(16, 8, 16, 8)
        
        controls_title = QLabel("Controls & Statistics")
        controls_title.setObjectName("panelTitle")
        controls_header_layout.addWidget(controls_title)
        
        self.detection_count_label = QLabel("Detections: 0")
        self.detection_count_label.setObjectName("statsLabel")
        controls_header_layout.addWidget(self.detection_count_label, 0, Qt.AlignRight)
        
        controls_layout.addWidget(controls_header)
        
        # Buttons container
        buttons_container = QWidget()
        buttons_container.setObjectName("buttonsContainer")
        buttons_layout = QHBoxLayout(buttons_container)
        buttons_layout.setContentsMargins(16, 16, 16, 16)
        buttons_layout.setSpacing(16)
        
        self.reset_button = QPushButton("Reset Statistics")
        self.reset_button.setObjectName("actionButton")
        self.reset_button.setIcon(QIcon("ui/resources/reset_icon.png"))
        self.reset_button.clicked.connect(self.reset_stats)
        buttons_layout.addWidget(self.reset_button)
        
        self.toggle_mode_button = QPushButton("Switch to ASL Mode")
        self.toggle_mode_button.setObjectName("primaryButton")
        self.toggle_mode_button.setIcon(QIcon("ui/resources/switch_icon.png"))
        self.toggle_mode_button.clicked.connect(self.toggle_mode)
        buttons_layout.addWidget(self.toggle_mode_button)
        
        controls_layout.addWidget(buttons_container)
        left_layout.addWidget(controls_panel)
        
        # Create right panel (results and history)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(16)
        
        # Result widget container
        result_container = QFrame()
        result_container.setObjectName("resultContainer")
        result_layout = QVBoxLayout(result_container)
        result_layout.setContentsMargins(0, 0, 0, 0)
        
        # Result header
        result_header = QWidget()
        result_header.setObjectName("panelHeader")
        result_header_layout = QHBoxLayout(result_header)
        result_header_layout.setContentsMargins(16, 8, 16, 8)
        
        result_title = QLabel("Detection Results")
        result_title.setObjectName("panelTitle")
        result_header_layout.addWidget(result_title)
        
        result_layout.addWidget(result_header)
        
        # Result widget
        self.result_widget = ResultWidget()
        result_layout.addWidget(self.result_widget)
        
        right_layout.addWidget(result_container)
        
        # History panel
        history_panel = QFrame()
        history_panel.setObjectName("historyPanel")
        history_layout = QVBoxLayout(history_panel)
        history_layout.setContentsMargins(0, 0, 0, 0)
        
        # History header
        history_header = QWidget()
        history_header.setObjectName("panelHeader")
        history_header_layout = QHBoxLayout(history_header)
        history_header_layout.setContentsMargins(16, 8, 16, 8)
        
        history_title = QLabel("Detection History")
        history_title.setObjectName("panelTitle")
        history_header_layout.addWidget(history_title)
        
        clear_history_btn = QPushButton("Clear")
        clear_history_btn.setObjectName("smallButton")
        clear_history_btn.clicked.connect(self.clear_history)
        history_header_layout.addWidget(clear_history_btn, 0, Qt.AlignRight)
        
        history_layout.addWidget(history_header)
        
        # History content
        history_content = QWidget()
        history_content.setObjectName("historyContent")
        history_content_layout = QVBoxLayout(history_content)
        history_content_layout.setContentsMargins(16, 16, 16, 16)
        
        self.history_label = QLabel()
        self.history_label.setObjectName("historyText")
        self.history_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.history_label.setWordWrap(True)
        history_content_layout.addWidget(self.history_label)
        
        history_layout.addWidget(history_content)
        right_layout.addWidget(history_panel)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 3)
        main_layout.addWidget(right_panel, 2)
        
        # Initialize timer for updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(50)  # 20 FPS refresh
        
        # Initialize gesture history
        self.gesture_history = GestureHistory(max_size=10)
        self.detection_count = 0
        self.mode = "gesture"  # "gesture" or "asl"
        
        # Connect camera widget signals
        self.camera_widget.gesture_detected.connect(self.on_gesture_detected)
        self.camera_widget.asl_detected.connect(self.on_asl_detected)
        self.camera_widget.fps_updated.connect(self.update_fps)
    
    def setup_styles(self):
        # Load fonts
        QFontDatabase.addApplicationFont("ui/resources/Inter-Regular.ttf")
        QFontDatabase.addApplicationFont("ui/resources/Inter-Medium.ttf")
        QFontDatabase.addApplicationFont("ui/resources/Inter-Bold.ttf")
        
        # Set app style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QWidget {
                background-color: transparent;
                color: #FFFFFF;
                font-family: 'Inter';
            }
            QFrame {
                background-color: #1E1E1E;
                border-radius: 12px;
            }
            #panelHeader {
                background-color: #252525;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }
            #panelTitle {
                font-size: 16px;
                font-weight: bold;
                color: #FFFFFF;
            }
            #statsLabel {
                color: #8AB4F8;
                font-size: 14px;
            }
            #buttonsContainer {
                background-color: transparent;
            }
            #historyContent {
                background-color: #252525;
                border-radius: 8px;
            }
            #historyText {
                color: #E0E0E0;
                font-size: 14px;
                line-height: 1.5;
            }
            QPushButton {
                border-radius: 6px;
                padding: 10px 16px;
                font-weight: medium;
                font-size: 14px;
            }
            #primaryButton {
                background-color: #8AB4F8;
                color: #121212;
                border: none;
            }
            #primaryButton:hover {
                background-color: #A5C8FF;
            }
            #actionButton {
                background-color: #3C4043;
                color: #FFFFFF;
                border: none;
            }
            #actionButton:hover {
                background-color: #4E5256;
            }
            #smallButton {
                background-color: #3C4043;
                color: #FFFFFF;
                border: none;
                padding: 4px 10px;
                font-size: 12px;
            }
            #smallButton:hover {
                background-color: #4E5256;
            }
        """)
    
    def update_ui(self):
        # Update history display
        history_text = ""
        for i, (item, timestamp) in enumerate(self.gesture_history.get_history()):
            history_text += f"{i+1}. {item} - {timestamp}\n"
        
        self.history_label.setText(history_text)
    
    @pyqtSlot(str)
    def on_gesture_detected(self, gesture):
        if self.mode == "gesture":
            self.result_widget.update_gesture(gesture)
            if gesture != "Unknown Gesture" and gesture != "No Hand Detected":
                self.gesture_history.add_item(gesture)
                self.detection_count += 1
                self.detection_count_label.setText(f"Detections: {self.detection_count}")
    
    @pyqtSlot(str, float)
    def on_asl_detected(self, letter, confidence):
        if self.mode == "asl" and letter is not None:
            self.result_widget.update_asl(letter, confidence)
            self.gesture_history.add_item(f"ASL: {letter} ({confidence:.2f})")
            self.detection_count += 1
            self.detection_count_label.setText(f"Detections: {self.detection_count}")
    
    @pyqtSlot(float)
    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    def reset_stats(self):
        self.detection_count = 0
        self.detection_count_label.setText("Detections: 0")
        self.clear_history()
    
    def clear_history(self):
        self.gesture_history.clear()
        self.history_label.setText("")
    
    def toggle_mode(self):
        if self.mode == "gesture":
            self.mode = "asl"
            self.toggle_mode_button.setText("Switch to Gesture Mode")
            self.result_widget.set_mode("asl")
        else:
            self.mode = "gesture"
            self.toggle_mode_button.setText("Switch to ASL Mode")
            self.result_widget.set_mode("gesture")
        self.camera_widget.set_mode(self.mode)
