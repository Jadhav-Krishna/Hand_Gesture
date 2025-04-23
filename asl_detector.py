import numpy as np
import cv2
from sklearn.preprocessing import normalize

class ASLDetector:
    def __init__(self):
        # ASL alphabet configurations
        self.asl_configs = self._initialize_asl_configurations()
        self.last_letter = None
        self.confidence_threshold = 0.65  # Lowered threshold for better detection
        self.stability_count = 0
        self.last_predictions = []
        self.prediction_history_size = 5
        
    def _initialize_asl_configurations(self):
        # This represents ASL letter configurations
        # Format: [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended, special_configuration]
        configs = {
            'A': [0.2, 0.1, 0.1, 0.1, 0.1, 'fist_with_thumb_side'],
            'B': [0.3, 0.9, 0.9, 0.9, 0.9, 'palm_forward'],
            'C': [0.5, 0.5, 0.5, 0.5, 0.5, 'curved_palm'],
            'D': [0.2, 0.9, 0.5, 0.1, 0.1, 'index_up_others_curved'],
            'E': [0.2, 0.2, 0.2, 0.2, 0.2, 'curved_fingers'],
            'F': [0.6, 0.6, 0.9, 0.9, 0.9, 'thumb_index_connect'],
            'G': [0.8, 0.9, 0.1, 0.1, 0.1, 'index_pointing_thumb_side'],
            'H': [0.8, 0.9, 0.9, 0.1, 0.1, 'index_middle_extended'],
            'I': [0.1, 0.1, 0.1, 0.1, 0.9, 'pinky_only'],
            'J': [0.1, 0.1, 0.1, 0.1, 0.9, 'pinky_only_moving'],
            'K': [0.8, 0.9, 0.7, 0.1, 0.1, 'index_middle_up_separated'],
            'L': [0.9, 0.9, 0.1, 0.1, 0.1, 'l_shape'],
            'M': [0.2, 0.3, 0.3, 0.3, 0.1, 'three_fingers_folded'],
            'N': [0.2, 0.3, 0.3, 0.1, 0.1, 'index_middle_folded'],
            'O': [0.6, 0.6, 0.6, 0.6, 0.6, 'o_shape'],
            'P': [0.8, 0.9, 0.1, 0.1, 0.1, 'p_shape'],
            'Q': [0.8, 0.9, 0.1, 0.1, 0.1, 'q_shape'],
            'R': [0.3, 0.9, 0.9, 0.1, 0.1, 'crossed_fingers'],
            'S': [0.2, 0.2, 0.2, 0.2, 0.2, 'fist_forward'],
            'T': [0.5, 0.3, 0.1, 0.1, 0.1, 'thumb_between_fingers'],
            'U': [0.3, 0.9, 0.9, 0.1, 0.1, 'u_shape'],
            'V': [0.3, 0.9, 0.9, 0.1, 0.1, 'peace_sign'],
            'W': [0.3, 0.9, 0.9, 0.9, 0.1, 'three_fingers_up'],
            'X': [0.3, 0.5, 0.1, 0.1, 0.1, 'hook_index'],
            'Y': [0.9, 0.1, 0.1, 0.1, 0.9, 'y_shape'],
            'Z': [0.3, 0.9, 0.1, 0.1, 0.1, 'z_motion']
        }
        return configs

    def detect_asl(self, landmarks, handedness="Right"):
        if landmarks is None or len(landmarks) == 0:
            return None, 0.0
        
        # Convert landmarks to numpy array if not already
        landmarks = np.array(landmarks)
        
        # Extract features for ASL detection
        features = self._extract_asl_features(landmarks)
        
        # Compare features with ASL configurations
        results = []
        
        for letter, config in self.asl_configs.items():
            confidence = self._calculate_asl_confidence(features, config, letter, handedness)
            results.append((letter, confidence))
        
        # Sort by confidence
        results.sort(key=lambda x: x[1], reverse=True)
        best_match, highest_confidence = results[0]
        
        # Add to prediction history for stability
        self.last_predictions.append(best_match)
        if len(self.last_predictions) > self.prediction_history_size:
            self.last_predictions.pop(0)
        
        # Only return a stable prediction
        if highest_confidence >= self.confidence_threshold:
            # Check if the prediction is stable
            if len(self.last_predictions) == self.prediction_history_size:
                most_common = max(set(self.last_predictions), key=self.last_predictions.count)
                if self.last_predictions.count(most_common) >= 3:  # At least 3 out of 5 predictions are the same
                    return most_common, highest_confidence
        
        return None, highest_confidence
    
    def _extract_asl_features(self, landmarks):
        # Convert landmarks to feature vector for ASL detection
        features = {}
        
        # Normalize landmarks to make them size and position invariant
        # First, center the landmarks around the wrist
        wrist = landmarks[0]
        centered_landmarks = landmarks - wrist
        
        # Scale by the hand size (distance from wrist to middle finger MCP)
        scale_reference = np.linalg.norm(centered_landmarks[9])
        if scale_reference > 0:
            normalized_landmarks = centered_landmarks / scale_reference
        else:
            normalized_landmarks = centered_landmarks
        
        # Base positions
        palm_center = np.mean(normalized_landmarks[[0, 5, 9, 13, 17]], axis=0)
        
        # Calculate finger extension values (0 to 1)
        features['thumb_extended'] = self._calculate_finger_extension(normalized_landmarks, 1, 2, 3, 4)
        features['index_extended'] = self._calculate_finger_extension(normalized_landmarks, 5, 6, 7, 8)
        features['middle_extended'] = self._calculate_finger_extension(normalized_landmarks, 9, 10, 11, 12)
        features['ring_extended'] = self._calculate_finger_extension(normalized_landmarks, 13, 14, 15, 16)
        features['pinky_extended'] = self._calculate_finger_extension(normalized_landmarks, 17, 18, 19, 20)
        
        # Calculate angles between fingers
        features['thumb_index_angle'] = self._angle_between_fingers(normalized_landmarks, 4, 0, 8)
        features['index_middle_angle'] = self._angle_between_fingers(normalized_landmarks, 8, 5, 12)
        features['middle_ring_angle'] = self._angle_between_fingers(normalized_landmarks, 12, 9, 16)
        features['ring_pinky_angle'] = self._angle_between_fingers(normalized_landmarks, 16, 13, 20)
        
        # Distances between fingertips
        features['thumb_index_dist'] = np.linalg.norm(normalized_landmarks[4] - normalized_landmarks[8])
        features['index_middle_dist'] = np.linalg.norm(normalized_landmarks[8] - normalized_landmarks[12])
        features['middle_ring_dist'] = np.linalg.norm(normalized_landmarks[12] - normalized_landmarks[16])
        features['ring_pinky_dist'] = np.linalg.norm(normalized_landmarks[16] - normalized_landmarks[20])
        
        # Fingertip to palm distances
        features['thumb_palm_dist'] = np.linalg.norm(normalized_landmarks[4] - palm_center)
        features['index_palm_dist'] = np.linalg.norm(normalized_landmarks[8] - palm_center)
        features['middle_palm_dist'] = np.linalg.norm(normalized_landmarks[12] - palm_center)
        features['ring_palm_dist'] = np.linalg.norm(normalized_landmarks[16] - palm_center)
        features['pinky_palm_dist'] = np.linalg.norm(normalized_landmarks[20] - palm_center)
        
        # Palm orientation approximation
        try:
            palm_normal = np.cross(
                normalized_landmarks[5] - normalized_landmarks[17],
                normalized_landmarks[0] - normalized_landmarks[9]
            )
            features['palm_orientation'] = palm_normal / (np.linalg.norm(palm_normal) + 1e-6)
        except:
            features['palm_orientation'] = np.array([0, 0, 1])
        
        # Store raw landmarks for special case detection
        features['landmarks'] = normalized_landmarks
        
        return features
    
    def _calculate_finger_extension(self, landmarks, mcp_idx, pip_idx, dip_idx, tip_idx):
        """
        Calculate how extended a finger is on a scale from 0 to 1
        0 = completely curled, 1 = completely extended
        """
        # Get vectors
        mcp_to_pip = landmarks[pip_idx] - landmarks[mcp_idx]
        pip_to_dip = landmarks[dip_idx] - landmarks[pip_idx]
        dip_to_tip = landmarks[tip_idx] - landmarks[dip_idx]
        
        # Normalize vectors
        try:
            mcp_to_pip = mcp_to_pip / (np.linalg.norm(mcp_to_pip) + 1e-6)
            pip_to_dip = pip_to_dip / (np.linalg.norm(pip_to_dip) + 1e-6)
            dip_to_tip = dip_to_tip / (np.linalg.norm(dip_to_tip) + 1e-6)
        except:
            return 0.0
        
        # Check alignment of the segments (dot products)
        alignment1 = np.dot(mcp_to_pip, pip_to_dip)
        alignment2 = np.dot(pip_to_dip, dip_to_tip)
        
        # Combine alignments to get extension value
        # If segments are well aligned, finger is more extended
        extension = (alignment1 + alignment2 + 2) / 4  # Map from [-1, 1] to [0, 1]
        return extension
    
    def _angle_between_fingers(self, landmarks, tip1_idx, base_idx, tip2_idx):
        """Calculate angle between two fingers"""
        try:
            v1 = landmarks[tip1_idx] - landmarks[base_idx]
            v2 = landmarks[tip2_idx] - landmarks[base_idx]
            
            v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
            v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)
            
            dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            angle = np.arccos(dot_product)
            return angle
        except:
            return 0.0
    
    def _calculate_asl_confidence(self, features, config, letter, handedness):
        """Calculate confidence score for an ASL letter"""
        # Weight factors for different feature categories
        finger_weight = 0.5
        angle_weight = 0.2
        distance_weight = 0.2
        special_weight = 0.1
        
        # Basic check for extended fingers
        finger_score = 0
        fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for i, finger in enumerate(fingers):
            # Use a softer comparison instead of binary check
            diff = abs(features[f'{finger}_extended'] - config[i])
            finger_score += max(0, 1 - diff * 2)  # Linear falloff
        finger_score /= len(fingers)  # Normalize to [0,1]
        
        # Angle score
        angle_score = 0
        # Only for letters where angle is important
        if letter in ['V', 'K', 'R', 'U', 'W', 'Y']:
            # Check specific angle features based on the letter
            if letter == 'V' or letter == 'U':
                # Small angle between index and middle
                angle = features['index_middle_angle']
                angle_score = max(0, 1 - angle)
            elif letter == 'K':
                # Wide angle between index and middle
                angle = features['index_middle_angle']
                angle_score = min(1, angle)
            elif letter == 'R':
                # Crossed index and middle
                angle_score = 1 if features['index_middle_dist'] < 0.2 else 0
            elif letter == 'W':
                # Three fingers with small angles
                angle_score = max(0, 1 - features['index_middle_angle']) * max(0, 1 - features['middle_ring_angle'])
            elif letter == 'Y':
                # Thumb and pinky extended
                thumb_pinky_angle = self._angle_between_fingers(features['landmarks'], 4, 0, 20)
                angle_score = min(1, thumb_pinky_angle)
        else:
            angle_score = 0.5  # Neutral for letters where angle is less important
        
        # Distance score
        distance_score = 0
        # Special distance checks for specific letters
        if letter == 'O':
            # Thumb and index tips should be close
            distance_score = max(0, 1 - features['thumb_index_dist'] * 3)
        elif letter == 'E':
            # All fingertips should be close to palm
            avg_dist = np.mean([features['thumb_palm_dist'], features['index_palm_dist'], 
                              features['middle_palm_dist'], features['ring_palm_dist'], 
                              features['pinky_palm_dist']])
            distance_score = max(0, 1 - avg_dist * 2)
        elif letter == 'S':
            # Fist - all fingers curled
            avg_dist = np.mean([features['thumb_palm_dist'], features['index_palm_dist'], 
                              features['middle_palm_dist'], features['ring_palm_dist'], 
                              features['pinky_palm_dist']])
            distance_score = max(0, 1 - avg_dist * 2)
        else:
            distance_score = 0.5  # Neutral for other letters
        
        # Special configuration handling
        special_score = 0
        letter_type = config[5]
        
        # Handle special cases for specific letters
        if letter_type == 'o_shape' and features['thumb_index_dist'] < 0.3:
            special_score += 1
        elif letter_type == 'l_shape' and features['thumb_index_angle'] > 0.8:
            special_score += 1
        elif letter_type == 'fist_with_thumb_side' and features['thumb_extended'] > 0.1 and features['index_extended'] < 0.3:
            special_score += 1
        elif letter_type == 'peace_sign' and features['index_extended'] > 0.7 and features['middle_extended'] > 0.7:
            special_score += 1
        elif letter_type == 'three_fingers_up' and (features['index_extended'] > 0.7 and 
                                                 features['middle_extended'] > 0.7 and 
                                                 features['ring_extended'] > 0.7):
            special_score += 1
        elif letter_type == 'y_shape' and features['thumb_extended'] > 0.7 and features['pinky_extended'] > 0.7:
            special_score += 1
        # Add more special cases here as needed
        else:
            special_score = 0.5  # Neutral score for letters without special handling
            
        # Handle handedness for asymmetric letters (like J and Z)
        if letter in ['J', 'Z'] and handedness != "Right":
            special_score *= 0.5  # Reduce score for wrong hand
        
        # Combine scores with weights
        total_score = (finger_score * finger_weight + 
                      angle_score * angle_weight + 
                      distance_score * distance_weight + 
                      special_score * special_weight)
        
        return total_score