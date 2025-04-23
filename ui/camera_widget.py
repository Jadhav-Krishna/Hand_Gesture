import cv2
import time
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

from gesture_detector import GestureDetector
from asl_detector import ASLDetector

class CameraWidget(QWidget):
    gesture_detected = pyqtSignal(str)
    asl_detected = pyqtSignal(str, float)
    fps_updated = pyqtSignal(float)
    
    def __init__(self):
        super().__init__()
        
        # Initialize layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create camera feed label
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("""
            border-radius: 8px;
            background-color: #121212;
        """)
        layout.addWidget(self.camera_label)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
        # Initialize detectors
        self.gesture_detector = GestureDetector()
        self.asl_detector = ASLDetector()
        
        # FPS calculation
        self.prev_frame_time = 0
        self.curr_frame_time = 0
        self.fps = 0
        
        # Mode selection
        self.mode = "gesture"  # "gesture" or "asl"
        
        # Set up timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 FPS
    
    def update_frame(self):
        ret, frame = self.cap.read()
        
        if not ret:
            return
        
        # Mirror the frame for more intuitive interaction
        frame = cv2.flip(frame, 1)
        
        # Calculate FPS
        self.curr_frame_time = time.time()
        self.fps = 1 / (self.curr_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 30
        self.prev_frame_time = self.curr_frame_time
        self.fps_updated.emit(self.fps)
        
        # Process frame based on current mode
        if self.mode == "gesture":
            self.process_gesture_mode(frame)
        else:  # asl mode
            self.process_asl_mode(frame)
    
    def process_gesture_mode(self, frame):
        # Detect hand gestures
        frame_with_landmarks, detected_gestures = self.gesture_detector.detect_gesture(frame)
        
        # Draw modern overlay
        frame_with_overlay = self.draw_modern_overlay(frame_with_landmarks)
        
        # Draw FPS counter
        cv2.putText(
            frame_with_overlay,
            f"FPS: {self.fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (138, 180, 248),  # Light blue
            2
        )
        
        # Draw mode indicator
        cv2.putText(
            frame_with_overlay,
            "MODE: GESTURE",
            (frame.shape[1] - 220, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (138, 180, 248),  # Light blue
            2
        )
        
        # Emit detected gesture signal (only first one if multiple hands detected)
        if detected_gestures:
            self.gesture_detected.emit(detected_gestures[0])
        else:
            self.gesture_detected.emit("No Hand Detected")
        
        # Convert to Qt format and display
        self.display_frame(frame_with_overlay)
    
    def process_asl_mode(self, frame):
        # Process ASL detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use MediaPipe for hand detection
        results = self.gesture_detector.hands.process(frame_rgb)
        frame_with_landmarks = frame.copy()
        
        # Draw landmarks if hands are detected
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks
                self.gesture_detector.mp_draw.draw_landmarks(
                    frame_with_landmarks, 
                    hand_landmarks, 
                    self.gesture_detector.mp_hands.HAND_CONNECTIONS,
                    self.gesture_detector.mp_draw.DrawingSpec(color=(138, 180, 248), thickness=2, circle_radius=4),
                    self.gesture_detector.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
                
                # Convert landmarks to numpy array
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                landmarks = np.array(landmarks)
                
                # Determine if it's a left or right hand
                handedness = results.multi_handedness[hand_idx].classification[0].label
                
                # Detect ASL letter
                letter, confidence = self.asl_detector.detect_asl(landmarks, handedness)
                
                # Emit ASL detection signal
                if letter:
                    self.asl_detected.emit(letter, confidence)
                
                # Draw detected letter on frame
                if letter:
                    cv2.putText(
                        frame_with_landmarks,
                        f"{letter} ({confidence:.2f})",
                        (int(hand_landmarks.landmark[0].x * frame.shape[1]), 
                         int(hand_landmarks.landmark[0].y * frame.shape[0]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (138, 180, 248),  # Light blue
                        2
                    )
        else:
            # No hands detected
            self.asl_detected.emit(None, 0.0)
        
        # Draw modern overlay
        frame_with_overlay = self.draw_modern_overlay(frame_with_landmarks)
        
        # Draw FPS counter
        cv2.putText(
            frame_with_overlay,
            f"FPS: {self.fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (138, 180, 248),  # Light blue
            2
        )
        
        # Draw mode indicator
        cv2.putText(
            frame_with_overlay,
            "MODE: ASL",
            (frame.shape[1] - 180, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (138, 180, 248),  # Light blue
            2
        )
        
        # Convert to Qt format and display
        self.display_frame(frame_with_overlay)
    
    def draw_modern_overlay(self, frame):
        height, width, _ = frame.shape
        overlay = frame.copy()
        
        # Draw rounded rectangle border
        # Since OpenCV doesn't have built-in rounded rectangle, we'll simulate it
        # with a semi-transparent overlay
        
        # Create a mask for rounded corners
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (width-10, height-10), 255, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 11)
        
        # Draw border
        border = overlay.copy()
        cv2.rectangle(border, (10, 10), (width-10, height-10), (138, 180, 248), 2)  # Light blue
        
        # Apply mask to border
        alpha = 0.7
        for c in range(3):
            overlay[:, :, c] = np.where(mask > 0, 
                                       overlay[:, :, c] * (1 - alpha) + border[:, :, c] * alpha,
                                       overlay[:, :, c])
        
        # Add subtle grid pattern
        grid_alpha = 0.1
        grid_spacing = 30
        grid_overlay = overlay.copy()
        
        # Horizontal grid lines
        for y in range(0, height, grid_spacing):
            cv2.line(grid_overlay, (0, y), (width, y), (138, 180, 248), 1)  # Light blue
            
        # Vertical grid lines
        for x in range(0, width, grid_spacing):
            cv2.line(grid_overlay, (x, 0), (x, height), (138, 180, 248), 1)  # Light blue
            
        # Apply grid with transparency
        cv2.addWeighted(grid_overlay, grid_alpha, overlay, 1 - grid_alpha, 0, overlay)
        
        # Add corner indicators (more subtle)
        corner_size = 15
        corner_color = (138, 180, 248)  # Light blue
        
        # Top-left
        cv2.line(overlay, (20, 20), (20 + corner_size, 20), corner_color, 2)
        cv2.line(overlay, (20, 20), (20, 20 + corner_size), corner_color, 2)
        
        # Top-right
        cv2.line(overlay, (width - 20, 20), (width - 20 - corner_size, 20), corner_color, 2)
        cv2.line(overlay, (width - 20, 20), (width - 20, 20 + corner_size), corner_color, 2)
        
        # Bottom-left
        cv2.line(overlay, (20, height - 20), (20 + corner_size, height - 20), corner_color, 2)
        cv2.line(overlay, (20, height - 20), (20, height - 20 - corner_size), corner_color, 2)
        
        # Bottom-right
        cv2.line(overlay, (width - 20, height - 20), (width - 20 - corner_size, height - 20), corner_color, 2)
        cv2.line(overlay, (width - 20, height - 20), (width - 20, height - 20 - corner_size), corner_color, 2)
        
        return overlay
    
    def display_frame(self, frame):
        # Convert OpenCV BGR image to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        
        # Convert to QImage
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit the label
        p = convert_to_qt_format.scaled(
            self.camera_label.width(), 
            self.camera_label.height(), 
            Qt.KeepAspectRatio
        )
        
        # Display on label
        self.camera_label.setPixmap(QPixmap.fromImage(p))
    
    def set_mode(self, mode):
        self.mode = mode
    
    def closeEvent(self, event):
        # Release the camera when the window is closed
        self.cap.release()
