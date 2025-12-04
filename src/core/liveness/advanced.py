"""
Advanced liveness detection using depth estimation.
Uses MiDaS model for monocular depth estimation to detect flat images/screens.
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DepthAnalysisResult:
    """Result of depth-based liveness analysis."""
    is_3d: bool
    confidence: float
    depth_map: Optional[np.ndarray]
    face_depth_variance: float
    nose_protrusion: float
    cheek_symmetry: float
    details: Dict


class DepthLivenessDetector:
    """
    Depth-based liveness detection using MiDaS.
    
    Detects spoofing attempts by analyzing:
    - Depth variance across face (flat = fake)
    - Nose protrusion (nose should be closer)
    - Cheek/forehead depth differences
    - Edge depth consistency
    """
    
    def __init__(
        self,
        model_type: str = "MiDaS_small",
        device: str = "cpu",
        depth_variance_threshold: float = 0.15,
        nose_protrusion_threshold: float = 0.05,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize depth liveness detector.
        
        Args:
            model_type: MiDaS model variant (MiDaS_small, DPT_Hybrid, DPT_Large)
            device: Device to run model on (cpu, cuda)
            depth_variance_threshold: Min depth variance for real face
            nose_protrusion_threshold: Min nose protrusion ratio
            confidence_threshold: Overall confidence threshold
        """
        self.model_type = model_type
        self.device = device
        self.depth_variance_threshold = depth_variance_threshold
        self.nose_protrusion_threshold = nose_protrusion_threshold
        self.confidence_threshold = confidence_threshold
        
        self.model = None
        self.transform = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Load MiDaS model."""
        try:
            import torch
            
            # Load MiDaS
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
            self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            
            if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
            
            self._initialized = True
            logger.info(f"MiDaS model loaded: {self.model_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load MiDaS: {e}")
            return False
    
    def estimate_depth(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth map for an image.
        
        Args:
            image: BGR image
            
        Returns:
            Depth map (higher = further)
        """
        if not self._initialized:
            if not self.initialize():
                return None
        
        try:
            import torch
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transform
            input_batch = self.transform(img_rgb).to(self.device)
            
            # Predict depth
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False
                ).squeeze()
            
            depth_map = prediction.cpu().numpy()
            
            # Normalize to 0-1
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            
            return depth_map
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return None
    
    def analyze_face_depth(
        self,
        face_image: np.ndarray,
        landmarks: Optional[Dict] = None
    ) -> DepthAnalysisResult:
        """
        Analyze face depth for liveness detection.
        
        Args:
            face_image: Cropped face image (BGR)
            landmarks: Optional facial landmarks with keys like 'nose', 'left_eye', etc.
            
        Returns:
            Depth analysis results
        """
        # Get depth map
        depth_map = self.estimate_depth(face_image)
        
        if depth_map is None:
            return DepthAnalysisResult(
                is_3d=False,
                confidence=0.0,
                depth_map=None,
                face_depth_variance=0.0,
                nose_protrusion=0.0,
                cheek_symmetry=0.0,
                details={"error": "Depth estimation failed"}
            )
        
        h, w = depth_map.shape
        
        # Calculate overall depth variance
        face_depth_variance = np.std(depth_map)
        
        # Analyze nose protrusion (center should be closer = lower depth value)
        center_region = depth_map[h//3:2*h//3, w//3:2*w//3]
        edge_region = np.concatenate([
            depth_map[:h//4, :].flatten(),
            depth_map[3*h//4:, :].flatten(),
            depth_map[:, :w//4].flatten(),
            depth_map[:, 3*w//4:].flatten()
        ])
        
        center_depth = np.mean(center_region)
        edge_depth = np.mean(edge_region)
        nose_protrusion = edge_depth - center_depth  # Positive if nose protrudes
        
        # Analyze cheek symmetry
        left_cheek = depth_map[h//3:2*h//3, w//6:w//3]
        right_cheek = depth_map[h//3:2*h//3, 2*w//3:5*w//6]
        cheek_symmetry = 1.0 - abs(np.mean(left_cheek) - np.mean(right_cheek))
        
        # Calculate forehead vs chin depth difference
        forehead = depth_map[:h//4, w//4:3*w//4]
        chin = depth_map[3*h//4:, w//4:3*w//4]
        vertical_variance = abs(np.mean(forehead) - np.mean(chin))
        
        # Score components
        variance_score = min(face_depth_variance / self.depth_variance_threshold, 1.0)
        protrusion_score = min(max(nose_protrusion, 0) / self.nose_protrusion_threshold, 1.0)
        symmetry_score = cheek_symmetry
        
        # Combined confidence
        confidence = (
            variance_score * 0.4 +
            protrusion_score * 0.4 +
            symmetry_score * 0.2
        )
        
        is_3d = confidence >= self.confidence_threshold
        
        return DepthAnalysisResult(
            is_3d=is_3d,
            confidence=confidence,
            depth_map=depth_map,
            face_depth_variance=face_depth_variance,
            nose_protrusion=nose_protrusion,
            cheek_symmetry=cheek_symmetry,
            details={
                "variance_score": variance_score,
                "protrusion_score": protrusion_score,
                "symmetry_score": symmetry_score,
                "vertical_variance": vertical_variance,
                "center_depth": center_depth,
                "edge_depth": edge_depth
            }
        )
    
    def visualize_depth(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        colormap: int = cv2.COLORMAP_MAGMA
    ) -> np.ndarray:
        """
        Create visualization of depth map overlaid on image.
        
        Args:
            image: Original BGR image
            depth_map: Normalized depth map (0-1)
            colormap: OpenCV colormap to use
            
        Returns:
            Visualization image
        """
        # Convert depth to colormap
        depth_colored = cv2.applyColorMap(
            (depth_map * 255).astype(np.uint8),
            colormap
        )
        
        # Resize to match image if needed
        if depth_colored.shape[:2] != image.shape[:2]:
            depth_colored = cv2.resize(depth_colored, (image.shape[1], image.shape[0]))
        
        # Blend with original
        blended = cv2.addWeighted(image, 0.5, depth_colored, 0.5, 0)
        
        return blended


class ChallengeResponseDetector:
    """
    Challenge-response based liveness detection.
    
    Presents random challenges and verifies responses:
    - Blink detection
    - Head turn (left/right)
    - Smile detection
    - Nod (up/down)
    """
    
    def __init__(
        self,
        challenge_timeout: float = 5.0,
        required_challenges: int = 2,
        max_attempts: int = 3
    ):
        """
        Initialize challenge-response detector.
        
        Args:
            challenge_timeout: Seconds to complete each challenge
            required_challenges: Number of challenges to pass
            max_attempts: Maximum attempts before failure
        """
        self.challenge_timeout = challenge_timeout
        self.required_challenges = required_challenges
        self.max_attempts = max_attempts
        
        self.challenges = [
            "blink",
            "turn_left",
            "turn_right",
            "smile",
            "open_mouth"
        ]
        
        # State tracking
        self._sessions: Dict[str, Dict] = {}
    
    def start_session(self, session_id: str) -> Dict:
        """
        Start a new challenge session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Session info with first challenge
        """
        import random
        import time
        
        # Select random challenges
        selected = random.sample(self.challenges, self.required_challenges)
        
        session = {
            "session_id": session_id,
            "challenges": selected,
            "current_index": 0,
            "completed": [],
            "start_time": time.time(),
            "attempts": 0,
            "status": "in_progress"
        }
        
        self._sessions[session_id] = session
        
        return {
            "session_id": session_id,
            "challenge": selected[0],
            "challenge_text": self._get_challenge_text(selected[0]),
            "timeout": self.challenge_timeout,
            "remaining": self.required_challenges
        }
    
    def _get_challenge_text(self, challenge: str) -> str:
        """Get human-readable challenge text."""
        texts = {
            "blink": "Please blink your eyes",
            "turn_left": "Please turn your head to the left",
            "turn_right": "Please turn your head to the right",
            "smile": "Please smile",
            "open_mouth": "Please open your mouth",
            "nod": "Please nod your head"
        }
        return texts.get(challenge, challenge)
    
    def check_response(
        self,
        session_id: str,
        face_data: Dict,
        landmarks: Dict
    ) -> Dict:
        """
        Check if the current challenge response is valid.
        
        Args:
            session_id: Session identifier
            face_data: Face detection data
            landmarks: Facial landmarks
            
        Returns:
            Challenge status and next challenge if passed
        """
        import time
        
        if session_id not in self._sessions:
            return {"error": "Session not found", "status": "failed"}
        
        session = self._sessions[session_id]
        
        if session["status"] != "in_progress":
            return {"status": session["status"]}
        
        # Check timeout
        elapsed = time.time() - session["start_time"]
        if elapsed > self.challenge_timeout:
            session["attempts"] += 1
            if session["attempts"] >= self.max_attempts:
                session["status"] = "failed"
                return {"status": "failed", "reason": "timeout"}
            
            # Reset timer for retry
            session["start_time"] = time.time()
            return {
                "status": "timeout",
                "attempts_remaining": self.max_attempts - session["attempts"],
                "challenge": session["challenges"][session["current_index"]]
            }
        
        # Check current challenge
        current_challenge = session["challenges"][session["current_index"]]
        passed = self._verify_challenge(current_challenge, face_data, landmarks)
        
        if passed:
            session["completed"].append(current_challenge)
            session["current_index"] += 1
            
            if session["current_index"] >= len(session["challenges"]):
                session["status"] = "passed"
                return {
                    "status": "passed",
                    "message": "All challenges completed successfully"
                }
            
            # Next challenge
            next_challenge = session["challenges"][session["current_index"]]
            session["start_time"] = time.time()
            
            return {
                "status": "next_challenge",
                "challenge": next_challenge,
                "challenge_text": self._get_challenge_text(next_challenge),
                "completed": len(session["completed"]),
                "remaining": len(session["challenges"]) - session["current_index"]
            }
        
        return {
            "status": "in_progress",
            "challenge": current_challenge,
            "time_remaining": self.challenge_timeout - elapsed
        }
    
    def _verify_challenge(
        self,
        challenge: str,
        face_data: Dict,
        landmarks: Dict
    ) -> bool:
        """
        Verify if a specific challenge was completed.
        
        Args:
            challenge: Challenge type
            face_data: Face detection data
            landmarks: Facial landmarks
            
        Returns:
            True if challenge passed
        """
        if challenge == "blink":
            return self._check_blink(landmarks)
        elif challenge == "turn_left":
            return self._check_head_turn(face_data, "left")
        elif challenge == "turn_right":
            return self._check_head_turn(face_data, "right")
        elif challenge == "smile":
            return self._check_smile(landmarks)
        elif challenge == "open_mouth":
            return self._check_mouth_open(landmarks)
        
        return False
    
    def _check_blink(self, landmarks: Dict) -> bool:
        """Check if eyes are closed (blinking)."""
        if not landmarks:
            return False
        
        # Calculate Eye Aspect Ratio
        left_eye = landmarks.get("left_eye", [])
        right_eye = landmarks.get("right_eye", [])
        
        if len(left_eye) < 6 or len(right_eye) < 6:
            return False
        
        def eye_aspect_ratio(eye):
            # Vertical distances
            v1 = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
            v2 = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
            # Horizontal distance
            h = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
            return (v1 + v2) / (2.0 * h + 1e-6)
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        return avg_ear < 0.2  # Eyes closed
    
    def _check_head_turn(self, face_data: Dict, direction: str) -> bool:
        """Check if head is turned in specified direction."""
        # Use face bounding box position relative to frame
        bbox = face_data.get("bbox", [])
        frame_width = face_data.get("frame_width", 640)
        
        if len(bbox) < 4:
            return False
        
        face_center_x = bbox[0] + bbox[2] / 2
        relative_x = face_center_x / frame_width
        
        if direction == "left":
            return relative_x < 0.35
        elif direction == "right":
            return relative_x > 0.65
        
        return False
    
    def _check_smile(self, landmarks: Dict) -> bool:
        """Check if person is smiling."""
        mouth = landmarks.get("mouth", [])
        
        if len(mouth) < 12:
            return False
        
        # Mouth width vs height ratio
        width = np.linalg.norm(np.array(mouth[0]) - np.array(mouth[6]))
        height = np.linalg.norm(np.array(mouth[3]) - np.array(mouth[9]))
        
        ratio = width / (height + 1e-6)
        
        return ratio > 2.5  # Wide smile
    
    def _check_mouth_open(self, landmarks: Dict) -> bool:
        """Check if mouth is open."""
        mouth = landmarks.get("mouth", [])
        
        if len(mouth) < 12:
            return False
        
        # Vertical mouth opening
        top_lip = np.mean([mouth[2], mouth[3], mouth[4]], axis=0)
        bottom_lip = np.mean([mouth[8], mouth[9], mouth[10]], axis=0)
        
        opening = np.linalg.norm(np.array(top_lip) - np.array(bottom_lip))
        mouth_width = np.linalg.norm(np.array(mouth[0]) - np.array(mouth[6]))
        
        ratio = opening / (mouth_width + 1e-6)
        
        return ratio > 0.3  # Mouth open
    
    def end_session(self, session_id: str) -> Dict:
        """End and clean up a session."""
        if session_id in self._sessions:
            result = self._sessions[session_id].copy()
            del self._sessions[session_id]
            return result
        return {"error": "Session not found"}


class Face3DAnalyzer:
    """
    3D face structure analysis for liveness detection.
    Uses facial landmarks to estimate 3D structure.
    """
    
    def __init__(
        self,
        flatness_threshold: float = 0.1,
        symmetry_threshold: float = 0.8
    ):
        """
        Initialize 3D face analyzer.
        
        Args:
            flatness_threshold: Max flatness score for real face
            symmetry_threshold: Min symmetry score for real face
        """
        self.flatness_threshold = flatness_threshold
        self.symmetry_threshold = symmetry_threshold
        
        # 3D reference face model (mean face)
        self.reference_3d_points = np.array([
            [0.0, 0.0, 0.0],          # Nose tip
            [0.0, -330.0, -65.0],     # Chin
            [-225.0, 170.0, -135.0],  # Left eye left corner
            [225.0, 170.0, -135.0],   # Right eye right corner
            [-150.0, -150.0, -125.0], # Left mouth corner
            [150.0, -150.0, -125.0]   # Right mouth corner
        ], dtype=np.float64)
    
    def analyze_3d_structure(
        self,
        image: np.ndarray,
        landmarks_2d: np.ndarray
    ) -> Dict:
        """
        Analyze 3D face structure from 2D landmarks.
        
        Args:
            image: Face image
            landmarks_2d: 2D facial landmarks (68 points or 6 key points)
            
        Returns:
            3D analysis results
        """
        h, w = image.shape[:2]
        
        # Camera matrix (approximate)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Assume no lens distortion
        dist_coeffs = np.zeros((4, 1))
        
        # Select key landmarks
        if len(landmarks_2d) >= 68:
            # Standard 68-point landmarks
            image_points = np.array([
                landmarks_2d[30],  # Nose tip
                landmarks_2d[8],   # Chin
                landmarks_2d[36],  # Left eye left corner
                landmarks_2d[45],  # Right eye right corner
                landmarks_2d[48],  # Left mouth corner
                landmarks_2d[54]   # Right mouth corner
            ], dtype=np.float64)
        elif len(landmarks_2d) >= 6:
            image_points = landmarks_2d[:6].astype(np.float64)
        else:
            return {
                "is_3d": False,
                "confidence": 0.0,
                "error": "Insufficient landmarks"
            }
        
        # Solve PnP to get pose
        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.reference_3d_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return {
                    "is_3d": False,
                    "confidence": 0.0,
                    "error": "PnP solve failed"
                }
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Get Euler angles
            euler_angles = self._rotation_matrix_to_euler(rotation_matrix)
            
            # Analyze pose for liveness
            pitch, yaw, roll = euler_angles
            
            # Flatness score (screens show limited rotation)
            flatness_score = abs(yaw) / 45.0 + abs(pitch) / 30.0
            flatness_score = min(flatness_score, 1.0)
            
            # Symmetry analysis
            left_points = image_points[[2, 4]]  # Left eye, left mouth
            right_points = image_points[[3, 5]]  # Right eye, right mouth
            
            left_dist = np.mean(np.abs(left_points[:, 0] - center[0]))
            right_dist = np.mean(np.abs(right_points[:, 0] - center[0]))
            
            symmetry_score = 1.0 - abs(left_dist - right_dist) / (w / 4)
            symmetry_score = max(0, min(symmetry_score, 1.0))
            
            # Depth estimation from translation
            depth = translation_vector[2][0]
            depth_confidence = min(abs(depth) / 500.0, 1.0)
            
            # Combined score
            confidence = (
                (1.0 - flatness_score) * 0.3 +
                symmetry_score * 0.3 +
                depth_confidence * 0.4
            )
            
            is_3d = confidence > 0.5
            
            return {
                "is_3d": is_3d,
                "confidence": confidence,
                "euler_angles": {
                    "pitch": float(pitch),
                    "yaw": float(yaw),
                    "roll": float(roll)
                },
                "flatness_score": float(flatness_score),
                "symmetry_score": float(symmetry_score),
                "depth": float(depth),
                "depth_confidence": float(depth_confidence)
            }
            
        except Exception as e:
            return {
                "is_3d": False,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _rotation_matrix_to_euler(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles (pitch, yaw, roll)."""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.degrees(x), np.degrees(y), np.degrees(z)
