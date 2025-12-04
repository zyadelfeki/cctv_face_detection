"""
Eye blink detection for liveness verification.
Uses facial landmarks to detect eye aspect ratio changes.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial import distance as dist
from loguru import logger


class EyeBlinkDetector:
    """Detect eye blinks using Eye Aspect Ratio (EAR) analysis."""
    
    # Eye landmark indices for dlib 68-point model
    LEFT_EYE_INDICES = list(range(36, 42))
    RIGHT_EYE_INDICES = list(range(42, 48))
    
    def __init__(
        self,
        ear_threshold: float = 0.25,
        consecutive_frames: int = 3,
        history_size: int = 30
    ):
        """
        Initialize eye blink detector.
        
        Args:
            ear_threshold: Eye Aspect Ratio threshold below which eye is considered closed
            consecutive_frames: Number of consecutive frames for blink confirmation
            history_size: Number of frames to keep in history for analysis
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.history_size = history_size
        
        # Per-face tracking
        self._ear_history: Dict[str, List[float]] = {}
        self._blink_counts: Dict[str, int] = {}
        self._frame_counters: Dict[str, int] = {}
    
    @staticmethod
    def compute_ear(eye_landmarks: np.ndarray) -> float:
        """
        Compute the Eye Aspect Ratio (EAR) for a single eye.
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Args:
            eye_landmarks: 6x2 array of eye landmark coordinates
            
        Returns:
            Eye aspect ratio value
        """
        if eye_landmarks.shape[0] != 6:
            return 0.5  # Default open eye value
        
        # Vertical distances
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Horizontal distance
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        if C == 0:
            return 0.5
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def compute_average_ear(
        self,
        landmarks: Dict[str, Tuple[int, int]]
    ) -> Tuple[float, float, float]:
        """
        Compute EAR for both eyes and return average.
        
        Args:
            landmarks: Dictionary of facial landmarks with keys like 'left_eye', 'right_eye'
            
        Returns:
            Tuple of (left_ear, right_ear, average_ear)
        """
        # Extract eye landmarks if available
        left_eye = landmarks.get('left_eye')
        right_eye = landmarks.get('right_eye')
        
        if left_eye is None or right_eye is None:
            # Try extracting from full 68-point landmarks
            if 'all' in landmarks:
                all_pts = np.array(landmarks['all'])
                left_eye = all_pts[self.LEFT_EYE_INDICES]
                right_eye = all_pts[self.RIGHT_EYE_INDICES]
            else:
                return 0.5, 0.5, 0.5
        
        left_eye = np.array(left_eye) if not isinstance(left_eye, np.ndarray) else left_eye
        right_eye = np.array(right_eye) if not isinstance(right_eye, np.ndarray) else right_eye
        
        left_ear = self.compute_ear(left_eye)
        right_ear = self.compute_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        return left_ear, right_ear, avg_ear
    
    def update(
        self,
        face_id: str,
        landmarks: Dict[str, Tuple[int, int]]
    ) -> Dict[str, any]:
        """
        Update blink detection state for a face.
        
        Args:
            face_id: Unique identifier for the face being tracked
            landmarks: Facial landmarks dictionary
            
        Returns:
            Dictionary with detection results
        """
        left_ear, right_ear, avg_ear = self.compute_average_ear(landmarks)
        
        # Initialize tracking for new face
        if face_id not in self._ear_history:
            self._ear_history[face_id] = []
            self._blink_counts[face_id] = 0
            self._frame_counters[face_id] = 0
        
        # Update history
        self._ear_history[face_id].append(avg_ear)
        if len(self._ear_history[face_id]) > self.history_size:
            self._ear_history[face_id].pop(0)
        
        # Check for blink
        blink_detected = False
        if avg_ear < self.ear_threshold:
            self._frame_counters[face_id] += 1
        else:
            if self._frame_counters[face_id] >= self.consecutive_frames:
                self._blink_counts[face_id] += 1
                blink_detected = True
            self._frame_counters[face_id] = 0
        
        return {
            'face_id': face_id,
            'left_ear': left_ear,
            'right_ear': right_ear,
            'average_ear': avg_ear,
            'blink_detected': blink_detected,
            'blink_count': self._blink_counts[face_id],
            'eye_closed': avg_ear < self.ear_threshold
        }
    
    def get_blink_count(self, face_id: str) -> int:
        """Get total blink count for a face."""
        return self._blink_counts.get(face_id, 0)
    
    def get_ear_variance(self, face_id: str) -> float:
        """
        Get EAR variance over history - low variance suggests static image.
        
        Args:
            face_id: Face identifier
            
        Returns:
            Variance of EAR values, or 0 if insufficient history
        """
        history = self._ear_history.get(face_id, [])
        if len(history) < 5:
            return 0.0
        return float(np.var(history))
    
    def is_likely_live(
        self,
        face_id: str,
        min_blinks: int = 1,
        min_variance: float = 0.001,
        observation_frames: int = 30
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Determine if face appears to be a live person based on blink analysis.
        
        Args:
            face_id: Face identifier
            min_blinks: Minimum blinks required in observation window
            min_variance: Minimum EAR variance expected from live person
            observation_frames: Number of frames to analyze
            
        Returns:
            Tuple of (is_live, analysis_details)
        """
        history = self._ear_history.get(face_id, [])
        blink_count = self._blink_counts.get(face_id, 0)
        variance = self.get_ear_variance(face_id)
        
        sufficient_data = len(history) >= observation_frames
        has_blinks = blink_count >= min_blinks
        has_variance = variance >= min_variance
        
        is_live = sufficient_data and (has_blinks or has_variance)
        
        return is_live, {
            'sufficient_data': sufficient_data,
            'blink_count': blink_count,
            'ear_variance': variance,
            'has_natural_movement': has_variance,
            'observation_frames': len(history),
            'confidence': min(1.0, (blink_count * 0.3 + variance * 100) / 2)
        }
    
    def reset(self, face_id: Optional[str] = None):
        """Reset tracking state for one or all faces."""
        if face_id:
            self._ear_history.pop(face_id, None)
            self._blink_counts.pop(face_id, None)
            self._frame_counters.pop(face_id, None)
        else:
            self._ear_history.clear()
            self._blink_counts.clear()
            self._frame_counters.clear()
