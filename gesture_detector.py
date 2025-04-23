import cv2
import mediapipe as mp
import numpy as np
import time

class GestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.last_gesture = None
        self.last_detection_time = 0
        self.gesture_stable_time = 0.5  # Time in seconds to confirm gesture

    def detect_gesture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        detected_gestures = []
        frame_with_landmarks = frame.copy()
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks on frame
                self.mp_draw.draw_landmarks(
                    frame_with_landmarks, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4),
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # Convert landmarks to numpy array
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                landmarks = np.array(landmarks)
                
                # Determine if it's a left or right hand
                handedness = results.multi_handedness[hand_idx].classification[0].label
                gesture = self._classify_gesture(landmarks, handedness)
                detected_gestures.append(gesture)
                
        return frame_with_landmarks, detected_gestures
    
    def _classify_gesture(self, landmarks, handedness):
        # Extract key points for gesture recognition
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        wrist = landmarks[0]
        thumb_mcp = landmarks[2]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        pinky_mcp = landmarks[17]
        
        # Calculate finger states (extended or not)
        extended_fingers = []
        
        # Thumb is special - compare it to index MCP for extension
        thumb_extended = self._is_thumb_extended(thumb_tip, thumb_mcp, wrist, handedness)
        extended_fingers.append(thumb_extended)
        
        # For other fingers, compare fingertip y coordinate to MCP y coordinate
        finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
        finger_mcps = [index_mcp, middle_mcp, ring_mcp, pinky_mcp]
        
        for tip, mcp in zip(finger_tips, finger_mcps):
            extended_fingers.append(tip[1] < mcp[1])
        
        # Recognize gestures based on extended fingers
        gesture = self._identify_gesture(extended_fingers, landmarks)
        
        return gesture
    
    def _is_thumb_extended(self, thumb_tip, thumb_mcp, wrist, handedness):
        if handedness == "Left":
            return thumb_tip[0] < thumb_mcp[0]
        else:  # Right hand
            return thumb_tip[0] > thumb_mcp[0]
    
    def _identify_gesture(self, extended_fingers, landmarks):
        thumb, index, middle, ring, pinky = extended_fingers
        
        # Get distances between fingertips
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        thumb_index_distance = np.linalg.norm(thumb_tip[:2] - index_tip[:2])
        
        # Get direction vectors for swipe detection
        wrist = landmarks[0]
        palm_center = np.mean(landmarks[[0, 5, 9, 13, 17]], axis=0)
        palm_direction = palm_center - wrist
        
        # Open Palm: all fingers extended
        if all(extended_fingers):
            if palm_direction[1] > 0:
                return "Palm Facing Down"
            else:
                return "Palm Facing Up"
                
        # Fist: no fingers extended
        if not any(extended_fingers):
            return "Fist"
            
        # Thumbs Up: only thumb extended, pointing up
        if thumb and not any(extended_fingers[1:]) and thumb_tip[1] < wrist[1]:
            return "Thumbs Up"
            
        # Thumbs Down: only thumb extended, pointing down
        if thumb and not any(extended_fingers[1:]) and thumb_tip[1] > wrist[1]:
            return "Thumbs Down"
            
        # Peace Sign: index and middle extended
        if not thumb and index and middle and not ring and not pinky:
            return "Peace Sign"
            
        # OK Sign: thumb and index form a circle
        if thumb and index and thumb_index_distance < 0.1:
            return "OK Sign"
            
        # Pointing: only index finger extended
        if not thumb and index and not middle and not ring and not pinky:
            return "Pointing"
            
        # Three Fingers Up: index, middle, ring extended
        if not thumb and index and middle and ring and not pinky:
            return "Three Fingers Up"
            
        # Rock Sign (I Love You): thumb, index, and pinky extended
        if thumb and index and not middle and not ring and pinky:
            return "Rock Sign"
            
        # Call Me Sign (Shaka): thumb and pinky extended
        if thumb and not index and not middle and not ring and pinky:
            return "Call Me Sign"
        
        # Default if no gesture is recognized
        return "Unknown Gesture"
        
    def draw_gesture_text(self, frame, gesture, position=(50, 50)):
        cv2.putText(
            frame,
            f"Gesture: {gesture}",
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        return frame