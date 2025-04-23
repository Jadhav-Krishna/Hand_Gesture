
Hand Gesture and ASL Recognition System
A modern, real-time hand gesture and American Sign Language (ASL) recognition application built with Python, OpenCV, MediaPipe, and PyQt5.

Features

Real-time Hand Gesture Recognition: Detects and classifies common hand gestures including:

- Open Palm
- Fist
- Thumbs Up/Down
- Peace Sign
- OK Sign
- Pointing
- Three Fingers Up
- Rock Sign (I Love You)
- Call Me Sign (Shaka)
- Palm Facing Up/Down

ASL Letter Detection: Recognizes American Sign Language alphabet letters with confidence scoring

Modern User Interface:

- Clean, material design-inspired dark theme
- Real-time camera feed with visual overlays
- Detection history tracking
- Confidence visualization for ASL detection
- Animated feedback for detected gestures

Dual Operation Modes: Switch between general gesture recognition and ASL letter detection

Technologies Used

- Python 3.8+
- OpenCV: Computer vision and image processing
- MediaPipe: Hand landmark detection
- PyQt5: Modern GUI framework
- NumPy: Numerical operations and data processing
- scikit-learn: Machine learning utilities for gesture classification

Installation

Prerequisites

- Python 3.8 or higher
- Webcam or camera device

Setup

Clone the repository:
```bash
git clone https://github.com/Jadhav-Krishna/Hand_Gesture.git
cd Hand_Gesture
```

Create and activate a virtual environment (recommended):
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```
Or install dependencies manually:
```bash
pip install opencv-python mediapipe numpy PyQt5 scikit-learn
```

Create necessary directories:
```bash
mkdir -p ui/resources
```

Download required font files to `ui/resources/` directory:
- Inter-Regular.ttf
- Inter-Medium.ttf
- Inter-Bold.ttf

Usage

Running the Application

Start the application by running:
```bash
python main.py
```

Using the Interface

- **Camera Feed**: The left panel shows the camera feed with hand landmarks when detected.
- **Detection Results**: The right panel displays the currently detected gesture or ASL letter.
- **Mode Switching**: Click the "Switch to ASL Mode" / "Switch to Gesture Mode" button to toggle between detection modes.
- **Statistics**: View detection count and FPS in the controls panel.
- **History**: Recent detections are displayed in the history panel at the bottom right.

Gestures

Position your hand in front of the camera and try different gestures:

- Open Palm: Hold your hand open with fingers extended
- Fist: Close your hand into a fist
- Thumbs Up/Down: Extend only your thumb up or down
- Peace Sign: Extend index and middle fingers in a V shape
- OK Sign: Form a circle with thumb and index finger
- Pointing: Extend only your index finger
- Three Fingers Up: Extend index, middle, and ring fingers
- Rock Sign: Extend thumb, index, and pinky fingers
- Call Me Sign: Extend thumb and pinky fingers

ASL Detection

In ASL mode, form ASL alphabet letters with your hand. The system will display the detected letter and confidence score.

Project Structure

hand-gesture-recognition/
├── main.py                  # Application entry point
├── gesture_detector.py      # Hand gesture detection logic
├── asl_detector.py          # ASL letter detection logic
├── ui/
│   ├── main_window.py       # Main application window
│   ├── camera_widget.py     # Camera feed and processing widget
│   ├── result_widget.py     # Results display widget
│   └── resources/           # Fonts and icons
├── utils/
│   └── gesture_history.py   # Gesture history tracking
└── requirements.txt         # Project dependencies

How It Works

Hand Detection: MediaPipe's hand tracking solution detects hand landmarks in the camera feed.
Feature Extraction: Key points from the hand landmarks are extracted and processed.

Gesture Classification:

- In gesture mode, the system analyzes finger positions and orientations to classify gestures.
- In ASL mode, the system compares hand configurations with predefined ASL letter patterns.

Visualization: Results are displayed in real-time with visual feedback.

Customization

Adding New Gestures

To add new gestures, modify the _identify_gesture method in gesture_detector.py:
```python
def _identify_gesture(self, extended_fingers, landmarks):
    # Add your custom gesture detection logic here
    if custom_condition:
        return "Custom Gesture Name"
```

Improving ASL Detection

To improve ASL detection accuracy, update the ASL configurations in asl_detector.py:
```python
def _initialize_asl_configurations(self):
    configs = {
        # Update or add new letter configurations
        'X': [0, 0.5, 0, 0, 0, 'improved_x_shape'],
    }
    return configs
```

Troubleshooting

Camera Not Working

- Ensure your webcam is properly connected
- Check if other applications are using the camera
- Try changing the camera index in camera_widget.py:
```python
self.cap = cv2.VideoCapture(1)  # Try different indices (0, 1, 2...)
```

Low Performance

- Reduce the camera resolution in camera_widget.py
- Decrease the frame update rate by modifying the timer interval
- Close other resource-intensive applications

Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

- Fork the repository
- Create your feature branch (`git checkout -b feature/amazing-feature`)
- Commit your changes (`git commit -m 'Add some amazing feature'`)
- Push to the branch (`git push origin feature/amazing-feature`)
- Open a Pull Request

License

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments

- MediaPipe for the hand tracking solution
- OpenCV for computer vision capabilities
- PyQt5 for the GUI framework

Created by Krishna Jadhav - krishanraviandrajadhav@gmail.com
