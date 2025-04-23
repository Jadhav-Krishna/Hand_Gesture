import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGraphicsDropShadowEffect, 
    QSizePolicy, QProgressBar, QHBoxLayout
)
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect, QSize, pyqtProperty
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QBrush, QPixmap

class AnimatedLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        
        # Set font and style
        font = QFont("Inter", 32)
        font.setBold(True)
        self.setFont(font)
        
        self.setStyleSheet("""
            color: #8AB4F8;
            background-color: transparent;
        """)
        
        # Animation properties
        self._scale = 1.0
    
    @pyqtProperty(float)
    def scale(self):
        return self._scale
    
    @scale.setter
    def scale(self, value):
        self._scale = value
        self.update()
    
    def animate_new_value(self):
        # Create scale animation
        self.anim = QPropertyAnimation(self, b"scale")
        self.anim.setDuration(300)
        self.anim.setStartValue(1.2)
        self.anim.setEndValue(1.0)
        self.anim.setEasingCurve(QEasingCurve.OutBack)
        self.anim.start()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        
        # Enable composition for proper rendering
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        
        # Scale the painting
        painter.translate(self.width() / 2, self.height() / 2)
        painter.scale(self._scale, self._scale)
        painter.translate(-self.width() / 2, -self.height() / 2)
        
        # Draw text using the parent's paintEvent
        super().paintEvent(event)

class ResultWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.current_gesture = "No Hand Detected"
        self.current_asl = None
        self.current_confidence = 0.0
        self.mode = "gesture"  # "gesture" or "asl"
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # Create main content area
        content_layout = QVBoxLayout()
        content_layout.setSpacing(24)
        
        # Create gesture visualization area
        vis_frame = QWidget()
        vis_frame.setObjectName("visualizationArea")
        vis_frame.setMinimumHeight(200)
        vis_frame.setStyleSheet("""
            #visualizationArea {
                background-color: #252525;
                border-radius: 8px;
            }
        """)
        
        vis_layout = QVBoxLayout(vis_frame)
        vis_layout.setContentsMargins(16, 16, 16, 16)
        
        # Gesture label
        self.gesture_label = AnimatedLabel("No Hand Detected")
        self.gesture_label.setMinimumHeight(100)
        vis_layout.addWidget(self.gesture_label)
        
        # Confidence bar (only shown in ASL mode)
        confidence_container = QWidget()
        confidence_layout = QVBoxLayout(confidence_container)
        confidence_layout.setContentsMargins(0, 8, 0, 0)
        
        self.confidence_label = QLabel("Confidence")
        self.confidence_label.setStyleSheet("color: #8AB4F8; font-size: 14px; font-weight: medium;")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setVisible(False)
        confidence_layout.addWidget(self.confidence_label)
        
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(True)
        self.confidence_bar.setFormat("%v%")
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                background-color: #2A2A2A;
                border: none;
                border-radius: 4px;
                text-align: center;
                color: white;
                height: 8px;
            }
            QProgressBar::chunk {
                background-color: #8AB4F8;
                border-radius: 4px;
            }
        """)
        self.confidence_bar.setVisible(False)
        confidence_layout.addWidget(self.confidence_bar)
        
        vis_layout.addWidget(confidence_container)
        
        content_layout.addWidget(vis_frame)
        
        # Description label
        self.description_label = QLabel(
            "Hold your hand in front of the camera to detect gestures."
        )
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("color: #E0E0E0; font-size: 14px; line-height: 1.4;")
        self.description_label.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(self.description_label)
        
        layout.addLayout(content_layout)
        
        # Info box
        info_frame = QWidget()
        info_frame.setObjectName("infoArea")
        info_frame.setStyleSheet("""
            #infoArea {
                background-color: #252525;
                border-radius: 8px;
            }
        """)
        
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(16, 16, 16, 16)
        
        info_header = QHBoxLayout()
        
        info_icon = QLabel("ℹ️")
        info_icon.setStyleSheet("font-size: 16px;")
        info_header.addWidget(info_icon)
        
        info_title = QLabel("USAGE INFORMATION")
        info_title.setStyleSheet("color: #8AB4F8; font-size: 14px; font-weight: bold;")
        info_header.addWidget(info_title)
        info_header.addStretch()
        
        info_layout.addLayout(info_header)
        
        self.info_content = QLabel(
            "Use various hand gestures to interact with the system. "
            "Try gestures like open palm, fist, thumbs up, peace sign, and more."
        )
        self.info_content.setWordWrap(True)
        self.info_content.setStyleSheet("color: #E0E0E0; font-size: 14px; line-height: 1.4; margin-top: 8px;")
        self.info_content.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        info_layout.addWidget(self.info_content)
        
        layout.addWidget(info_frame)
    
    def update_gesture(self, gesture):
        if gesture != self.current_gesture:
            self.current_gesture = gesture
            
            # Update label and animate
            self.gesture_label.setText(gesture)
            self.gesture_label.animate_new_value()
            
            # Update description based on gesture
            self.update_description(gesture)
    
    def update_asl(self, letter, confidence):
        if letter != self.current_asl or abs(confidence - self.current_confidence) > 0.1:
            self.current_asl = letter
            self.current_confidence = confidence
            
            # Update label and animate
            self.gesture_label.setText(letter if letter else "No ASL Detected")
            self.gesture_label.animate_new_value()
            
            # Update confidence bar
            self.confidence_bar.setValue(int(confidence * 100))
            
            # Update description for ASL letter
            self.update_asl_description(letter)
    
    def update_description(self, gesture):
        descriptions = {
            "Open Palm": "Open palm gesture detected. This can be used for navigation or selection.",
            "Fist": "Fist gesture detected. This can be used to grab or hold virtual objects.",
            "Thumbs Up": "Thumbs up gesture detected. This indicates approval or confirmation.",
            "Thumbs Down": "Thumbs down gesture detected. This indicates disapproval or rejection.",
            "Peace Sign": "Peace sign detected. This can be used for selection or navigation.",
            "OK Sign": "OK sign detected. This confirms selection or acceptance.",
            "Pointing": "Pointing gesture detected. Use this to indicate direction or select objects.",
            "Three Fingers Up": "Three fingers detected. This can trigger special functions.",
            "Rock Sign": "Rock sign (I Love You) detected. This can be used for specific commands.",
            "Call Me Sign": "Call Me (Shaka) sign detected.",
            "Palm Facing Down": "Palm facing down detected. This can be used for lowering or minimizing.",
            "Palm Facing Up": "Palm facing up detected. This can be used for raising or maximizing.",
            "No Hand Detected": "Hold your hand in front of the camera to detect gestures.",
            "Unknown Gesture": "Unrecognized gesture. Try one of the supported gestures."
        }
        
        self.description_label.setText(descriptions.get(gesture, "Custom gesture detected."))
    
    def update_asl_description(self, letter):
        if letter:
            self.description_label.setText(f"ASL Letter '{letter}' detected with confidence score.")
        else:
            self.description_label.setText("Hold your hand in front of the camera to detect ASL letters.")
    
    def set_mode(self, mode):
        self.mode = mode
        
        if mode == "gesture":
            self.confidence_label.setVisible(False)
            self.confidence_bar.setVisible(False)
            self.info_content.setText(
                "Use various hand gestures to interact with the system. "
                "Try gestures like open palm, fist, thumbs up, peace sign, and more."
            )
        else:  # asl mode
            self.confidence_label.setVisible(True)
            self.confidence_bar.setVisible(True)
            self.info_content.setText(
                "Show American Sign Language (ASL) letters with your hand. "
                "Position your hand clearly in front of the camera for best results."
            )
