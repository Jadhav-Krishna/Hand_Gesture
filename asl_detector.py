import numpy as np
import cv2
from sklearn.preprocessing import normalize

class ASLDetector:
    def __init__(self):
        # ASL alphabet configurations (simplified for demonstration)
        # These will be compared with detected hand landmarks
        self.asl_configs = self._initialize_asl_configurations()
        self.last_letter = None
        self.confidence_threshold = 0.75
        
    def _initialize_asl_configurations(self):
        # This is a simplified representation of ASL configurations
        # In a real app, these would be more detailed or learned from data
        configs = {
            # Basic relative positions of key points for each letter
            # Format: [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended, ...]
            'A': [0, 0, 0, 0, 0, 'fist_with_thumb_side'],
            'B': [0, 1, 1, 1, 1, 'palm_forward'],
            'C': [0.5, 0.5, 0.5, 0.5, 0.5, 'curved_palm'],
            'D': [0, 1, 0.5, 0, 0, 'index_up_others_curved'],
            'E': [0, 0, 0, 0, 0, 'curved_fingers'],
            'F': [0.5, 0.5, 1, 1, 1, 'thumb_index_connect'],
            'G': [1, 1, 0, 0, 0, 'index_pointing_thumb_side'],
            'H': [1, 1, 1, 0, 0, 'index_middle_extended'],
            'I': [0, 0, 0, 0, 1, 'pinky_only'],
            'J': [0, 0, 0, 0, 1, 'pinky_only_moving'],
            'K': [1, 1, 0, 0, 0, 'index_middle_up_separated'],
            'L': [1, 1, 0, 0, 0, 'l_shape'],
            'M': [0, 0, 0, 0, 0, 'three_fingers_folded'],
            'N': [0, 0, 0, 0, 0, 'index_middle_folded'],
            'O': [0.5, 0.5, 0.5, 0.5, 0.5, 'o_shape'],
            'P': [1, 1, 0, 0, 0, 'p_shape'],
            'Q': [1, 1, 0, 0, 0, 'q_shape'],
            'R': [0, 1, 1, 0, 0, 'crossed_fingers'],
            'S': [0, 0, 0, 0, 0, 'fist_forward'],
            'T': [0.5, 0, 0, 0, 0, 'thumb_between_fingers'],
            'U': [0, 1, 1, 0, 0, 'u_shape'],
            'V': [0, 1, 1, 0, 0, 'peace_sign'],
            'W': [0, 1, 1, 1, 0, 'three_fingers_up'],
            'X': [0, 0.5, 0, 0, 0, 'hook_index'],
            'Y': [1, 0, 0, 0, 1, 'y_shape'],
            'Z': [0, 1, 0, 0, 0, 'z_motion']
        }
        return configs

    def detect_asl(self, landmarks, handedness="Right"):
        if landmarks is None or len(landmarks) == 0:
            return None, 0.0
        
        # Extract features for ASL detection
        features = self._extract_asl_features(landmarks)
        
        # Compare features with ASL configurations
        best_match = None
        highest_confidence = 0.0
        
        for letter, config in self.asl_configs.items():
            confidence = self._calculate_asl_confidence(features, config, handedness)
            
            if confidence > highest_confidence:
                highest_confidence = confidence
                best_match = letter
        
        if highest_confidence < self.confidence_threshold:
            return None, highest_confidence
        
        return best_match, highest_confidence
    
    def _extract_asl_features(self, landmarks):
        # Convert landmarks to feature vector for ASL detection
        features = {}
        
        # Base positions
        wrist = landmarks[0]
        palm_center = np.mean(landmarks[[0, 5, 9, 13, 17]], axis=0)
        
        # Normalized finger positions
        thumb_tip_norm = landmarks[4] - wrist
        index_tip_norm = landmarks[8] - wrist
        middle_tip_norm = landmarks[12] - wrist
        ring_tip_norm = landmarks[16] - wrist
        pinky_tip_norm = landmarks[20] - wrist
        
        # Extended or bent fingers
        features['thumb_extended'] = self._is_finger_extended(landmarks[4], landmarks[3], landmarks[2])
        features['index_extended'] = self._is_finger_extended(landmarks[8], landmarks[6], landmarks[5])
        features['middle_extended'] = self._is_finger_extended(landmarks[12], landmarks[10], landmarks[9])
        features['ring_extended'] = self._is_finger_extended(landmarks[16], landmarks[14], landmarks[13])
        features['pinky_extended'] = self._is_finger_extended(landmarks[20], landmarks[18], landmarks[17])
        
        # Angles between fingers
        features['thumb_index_angle'] = self._angle_between(thumb_tip_norm, index_tip_norm)
        features['index_middle_angle'] = self._angle_between(index_tip_norm, middle_tip_norm)
        features['middle_ring_angle'] = self._angle_between(middle_tip_norm, ring_tip_norm)
        features['ring_pinky_angle'] = self._angle_between(ring_tip_norm, pinky_tip_norm)
        
        # Distances between fingertips
        features['thumb_index_dist'] = np.linalg.norm(landmarks[4] - landmarks[8])
        features['index_middle_dist'] = np.linalg.norm(landmarks[8] - landmarks[12])
        features['middle_ring_dist'] = np.linalg.norm(landmarks[12] - landmarks[16])
        features['ring_pinky_dist'] = np.linalg.norm(landmarks[16] - landmarks[20])
        
        # Palm orientation features
        palm_normal = np.cross(
            landmarks[5] - landmarks[17],
            landmarks[0] - landmarks[9]
        )
        features['palm_orientation'] = palm_normal / np.linalg.norm(palm_normal)
        
        return features
    
    def _is_finger_extended(self, tip, mid, mcp):
        # Determine if a finger is extended based on angles
        vec1 = tip - mid
        vec2 = mid - mcp
        
        # Normalize vectors
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        
        # Calculate dot product
        dot_product = np.dot(vec1, vec2)
        
        # If fingers are aligned (close to 1), the finger is extended
        return dot_product > 0.7
    
    def _angle_between(self, v1, v2):
        # Calculate angle between two vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        dot_product = np.dot(v1_norm, v2_norm)
        dot_product = max(min(dot_product, 1.0), -1.0)  # Clamp to avoid numerical errors
        return np.arccos(dot_product)
    
    def _calculate_asl_confidence(self, features, config, handedness):
        # Simple confidence calculation - can be improved significantly
        # In a real app, you'd use a proper classifier or neural network
        
        # Basic check for extended fingers
        finger_score = 0
        fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for i, finger in enumerate(fingers):
            if abs(features[f'{finger}_extended'] - config[i]) < 0.5:
                finger_score += 1
        
        # Special case handling for specific letters
        letter_type = config[5]
        special_score = 0
        
        if letter_type == 'o_shape' and features['thumb_index_dist'] < 0.1:
            special_score += 1
        elif letter_type == 'l_shape' and features['thumb_index_angle'] > 1.0:
            special_score += 1
        # Add more special cases for other letters
        
        # Combine scores
        total_score = (finger_score / 5.0) * 0.7 + (special_score / 1.0) * 0.3
        return total_score