"""
Main liveness detection module combining multiple anti-spoofing techniques.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import cv2
from loguru import logger

from .eye_blink import EyeBlinkDetector
from .texture import TextureAnalyzer


class LivenessDetector:
    """
    Comprehensive liveness detection combining multiple techniques:
    - Eye blink detection (requires video frames)
    - Texture analysis (LBP, frequency analysis)
    - Color analysis
    - Reflection detection
    
    Works with both single images and video streams.
    """
    
    def __init__(
        self,
        config=None,
        # Eye blink parameters
        ear_threshold: float = 0.25,
        min_blinks: int = 1,
        blink_observation_frames: int = 30,
        # Texture parameters
        lbp_radius: int = 1,
        frequency_threshold: float = 0.15,
        reflection_threshold: float = 0.3,
        # Overall thresholds
        liveness_threshold: float = 0.6,
        require_blink: bool = False
    ):
        """
        Initialize liveness detector.
        
        Args:
            config: Optional configuration object
            ear_threshold: Eye aspect ratio threshold for blink detection
            min_blinks: Minimum blinks required for liveness
            blink_observation_frames: Frames to observe for blink detection
            lbp_radius: LBP neighborhood radius
            frequency_threshold: Threshold for screen detection
            reflection_threshold: Threshold for reflection detection
            liveness_threshold: Overall threshold for liveness decision
            require_blink: If True, blink is required for liveness (video only)
        """
        self.config = config
        self.liveness_threshold = liveness_threshold
        self.require_blink = require_blink
        self.min_blinks = min_blinks
        self.blink_observation_frames = blink_observation_frames
        
        # Initialize sub-detectors
        self.eye_blink = EyeBlinkDetector(
            ear_threshold=ear_threshold,
            consecutive_frames=3,
            history_size=blink_observation_frames
        )
        
        self.texture = TextureAnalyzer(
            lbp_radius=lbp_radius,
            frequency_threshold=frequency_threshold,
            reflection_threshold=reflection_threshold
        )
        
        # Tracking for video mode
        self._face_states: Dict[str, Dict] = {}
    
    def check_single_image(
        self,
        face_image: np.ndarray,
        landmarks: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Check liveness on a single image (limited accuracy).
        
        Note: Single image analysis cannot use blink detection.
        For best results, use video-based detection.
        
        Args:
            face_image: Cropped face image (BGR)
            landmarks: Optional facial landmarks
            
        Returns:
            Liveness analysis results
        """
        if face_image is None or face_image.size == 0:
            return {
                'is_live': False,
                'confidence': 0.0,
                'error': 'Invalid image',
                'mode': 'single_image'
            }
        
        # Texture analysis
        texture_result = self.texture.analyze(face_image)
        
        # For single image, we rely entirely on texture analysis
        is_live = texture_result.get('is_live', False)
        confidence = texture_result.get('confidence', 0.0)
        
        return {
            'is_live': is_live,
            'confidence': confidence,
            'mode': 'single_image',
            'texture_analysis': texture_result,
            'blink_analysis': None,  # Not available for single image
            'warning': 'Single image mode has limited accuracy. Video mode recommended.'
        }
    
    def process_frame(
        self,
        face_id: str,
        face_image: np.ndarray,
        landmarks: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Process a video frame for liveness detection.
        
        Call this repeatedly for each frame to build up blink detection data.
        
        Args:
            face_id: Unique identifier for the face being tracked
            face_image: Cropped face image (BGR)
            landmarks: Facial landmarks for blink detection
            
        Returns:
            Current liveness analysis results
        """
        if face_image is None or face_image.size == 0:
            return {
                'face_id': face_id,
                'is_live': False,
                'confidence': 0.0,
                'error': 'Invalid image'
            }
        
        # Initialize state for new face
        if face_id not in self._face_states:
            self._face_states[face_id] = {
                'frames_analyzed': 0,
                'texture_scores': [],
                'last_texture_result': None
            }
        
        state = self._face_states[face_id]
        state['frames_analyzed'] += 1
        
        # Texture analysis (do periodically, not every frame)
        if state['frames_analyzed'] % 5 == 1:
            texture_result = self.texture.analyze(face_image)
            state['texture_scores'].append(texture_result.get('confidence', 0.0))
            state['last_texture_result'] = texture_result
            # Keep only recent scores
            if len(state['texture_scores']) > 20:
                state['texture_scores'].pop(0)
        
        texture_result = state.get('last_texture_result', {})
        
        # Blink analysis (if landmarks provided)
        blink_result = None
        is_live_blink = None
        blink_details = None
        
        if landmarks:
            blink_update = self.eye_blink.update(face_id, landmarks)
            is_live_blink, blink_details = self.eye_blink.is_likely_live(
                face_id,
                min_blinks=self.min_blinks,
                observation_frames=self.blink_observation_frames
            )
            blink_result = {
                'current': blink_update,
                'is_live': is_live_blink,
                'details': blink_details
            }
        
        # Combine scores
        texture_confidence = np.mean(state['texture_scores']) if state['texture_scores'] else 0.5
        
        if blink_result and blink_details:
            blink_confidence = blink_details.get('confidence', 0.5)
            
            if self.require_blink:
                # Both texture and blink must pass
                combined_confidence = min(texture_confidence, blink_confidence)
                is_live = texture_result.get('is_live', False) and is_live_blink
            else:
                # Weighted average (texture is more reliable for photo attacks)
                combined_confidence = texture_confidence * 0.6 + blink_confidence * 0.4
                is_live = combined_confidence >= self.liveness_threshold
        else:
            combined_confidence = texture_confidence
            is_live = texture_result.get('is_live', False) and combined_confidence >= self.liveness_threshold
        
        return {
            'face_id': face_id,
            'is_live': is_live,
            'confidence': float(combined_confidence),
            'frames_analyzed': state['frames_analyzed'],
            'mode': 'video',
            'texture_analysis': texture_result,
            'blink_analysis': blink_result,
            'sufficient_data': state['frames_analyzed'] >= self.blink_observation_frames
        }
    
    def get_final_result(self, face_id: str) -> Dict[str, any]:
        """
        Get final liveness result after processing multiple frames.
        
        Args:
            face_id: Face identifier
            
        Returns:
            Final liveness determination
        """
        state = self._face_states.get(face_id)
        if not state:
            return {
                'face_id': face_id,
                'is_live': False,
                'confidence': 0.0,
                'error': 'Face not tracked'
            }
        
        texture_confidence = np.mean(state['texture_scores']) if state['texture_scores'] else 0.0
        
        is_live_blink, blink_details = self.eye_blink.is_likely_live(
            face_id,
            min_blinks=self.min_blinks,
            observation_frames=self.blink_observation_frames
        )
        
        blink_confidence = blink_details.get('confidence', 0.0) if blink_details else 0.0
        
        if self.require_blink:
            combined_confidence = min(texture_confidence, blink_confidence)
            is_live = combined_confidence >= self.liveness_threshold and is_live_blink
        else:
            combined_confidence = texture_confidence * 0.6 + blink_confidence * 0.4
            is_live = combined_confidence >= self.liveness_threshold
        
        return {
            'face_id': face_id,
            'is_live': is_live,
            'confidence': float(combined_confidence),
            'texture_confidence': float(texture_confidence),
            'blink_confidence': float(blink_confidence),
            'blink_count': self.eye_blink.get_blink_count(face_id),
            'frames_analyzed': state['frames_analyzed'],
            'determination': 'LIVE' if is_live else 'SPOOF'
        }
    
    def reset(self, face_id: Optional[str] = None):
        """Reset tracking state."""
        if face_id:
            self._face_states.pop(face_id, None)
            self.eye_blink.reset(face_id)
        else:
            self._face_states.clear()
            self.eye_blink.reset()


class EnhancedLivenessDetector(LivenessDetector):
    """
    Enhanced liveness detector with advanced anti-spoofing techniques.
    
    Adds:
    - MiDaS depth estimation
    - Challenge-response verification
    - 3D face structure analysis
    """
    
    def __init__(
        self,
        use_depth: bool = True,
        use_3d_analysis: bool = True,
        use_challenge_response: bool = False,
        depth_weight: float = 0.3,
        **kwargs
    ):
        """
        Initialize enhanced liveness detector.
        
        Args:
            use_depth: Enable MiDaS depth analysis
            use_3d_analysis: Enable 3D face structure analysis
            use_challenge_response: Enable challenge-response mode
            depth_weight: Weight for depth analysis in final score
            **kwargs: Arguments passed to base LivenessDetector
        """
        super().__init__(**kwargs)
        
        self.use_depth = use_depth
        self.use_3d_analysis = use_3d_analysis
        self.use_challenge_response = use_challenge_response
        self.depth_weight = depth_weight
        
        # Initialize advanced detectors
        self._depth_detector = None
        self._3d_analyzer = None
        self._challenge_detector = None
        
        if use_depth:
            from .advanced import DepthLivenessDetector
            self._depth_detector = DepthLivenessDetector()
        
        if use_3d_analysis:
            from .advanced import Face3DAnalyzer
            self._3d_analyzer = Face3DAnalyzer()
        
        if use_challenge_response:
            from .advanced import ChallengeResponseDetector
            self._challenge_detector = ChallengeResponseDetector()
    
    def check_single_image_enhanced(
        self,
        face_image: np.ndarray,
        landmarks: Optional[Dict] = None,
        landmarks_2d: Optional[np.ndarray] = None
    ) -> Dict[str, any]:
        """
        Enhanced single image liveness check with depth and 3D analysis.
        
        Args:
            face_image: Cropped face image (BGR)
            landmarks: Optional facial landmarks dict
            landmarks_2d: Optional 2D landmarks array for 3D analysis
            
        Returns:
            Enhanced liveness analysis results
        """
        # Base texture analysis
        base_result = self.check_single_image(face_image, landmarks)
        base_confidence = base_result.get('confidence', 0.0)
        
        results = {
            'texture_analysis': base_result,
            'depth_analysis': None,
            '3d_analysis': None
        }
        
        scores = [base_confidence]
        weights = [0.4]
        
        # Depth analysis
        if self.use_depth and self._depth_detector:
            depth_result = self._depth_detector.analyze_face_depth(face_image, landmarks)
            results['depth_analysis'] = {
                'is_3d': depth_result.is_3d,
                'confidence': depth_result.confidence,
                'face_depth_variance': depth_result.face_depth_variance,
                'nose_protrusion': depth_result.nose_protrusion
            }
            scores.append(depth_result.confidence)
            weights.append(self.depth_weight)
        
        # 3D structure analysis
        if self.use_3d_analysis and self._3d_analyzer and landmarks_2d is not None:
            analysis_3d = self._3d_analyzer.analyze_3d_structure(face_image, landmarks_2d)
            results['3d_analysis'] = analysis_3d
            if 'confidence' in analysis_3d:
                scores.append(analysis_3d['confidence'])
                weights.append(0.3)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Combined confidence
        combined_confidence = sum(s * w for s, w in zip(scores, weights))
        is_live = combined_confidence >= self.liveness_threshold
        
        return {
            'is_live': is_live,
            'confidence': float(combined_confidence),
            'mode': 'single_image_enhanced',
            **results,
            'component_scores': {
                'texture': base_confidence,
                'depth': results['depth_analysis']['confidence'] if results['depth_analysis'] else None,
                '3d': results['3d_analysis'].get('confidence') if results['3d_analysis'] else None
            }
        }
    
    def start_challenge_session(self, session_id: str) -> Dict:
        """Start a challenge-response session."""
        if not self._challenge_detector:
            return {'error': 'Challenge-response not enabled'}
        return self._challenge_detector.start_session(session_id)
    
    def process_challenge_response(
        self,
        session_id: str,
        face_data: Dict,
        landmarks: Dict
    ) -> Dict:
        """Process a challenge response."""
        if not self._challenge_detector:
            return {'error': 'Challenge-response not enabled'}
        return self._challenge_detector.check_response(session_id, face_data, landmarks)
    
    def get_depth_visualization(
        self,
        face_image: np.ndarray
    ) -> Optional[np.ndarray]:
        """Get depth visualization for debugging."""
        if not self._depth_detector:
            return None
        
        depth_result = self._depth_detector.analyze_face_depth(face_image)
        if depth_result.depth_map is not None:
            return self._depth_detector.visualize_depth(
                face_image,
                depth_result.depth_map
            )
        return None
