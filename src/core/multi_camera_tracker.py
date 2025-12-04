"""
Multi-camera person tracking and re-identification module.
Tracks individuals across multiple camera feeds with timeline visualization.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import uuid
import heapq
import logging

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection from a camera."""
    detection_id: str
    camera_id: str
    timestamp: datetime
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    embedding: np.ndarray
    confidence: float
    frame_number: int
    snapshot_path: Optional[str] = None


@dataclass
class Track:
    """Track of a person within a single camera."""
    track_id: str
    camera_id: str
    person_id: Optional[str] = None  # Cross-camera ID
    detections: List[Detection] = field(default_factory=list)
    state: str = "active"  # active, lost, ended
    last_seen: Optional[datetime] = None
    embedding_history: List[np.ndarray] = field(default_factory=list)
    
    @property
    def mean_embedding(self) -> Optional[np.ndarray]:
        """Get mean embedding for this track."""
        if self.embedding_history:
            return np.mean(self.embedding_history, axis=0)
        return None
    
    @property
    def duration(self) -> timedelta:
        """Get track duration."""
        if len(self.detections) < 2:
            return timedelta(0)
        return self.detections[-1].timestamp - self.detections[0].timestamp
    
    def add_detection(self, detection: Detection):
        """Add detection to track."""
        self.detections.append(detection)
        self.embedding_history.append(detection.embedding)
        self.last_seen = detection.timestamp
        
        # Keep embedding history manageable
        if len(self.embedding_history) > 50:
            self.embedding_history = self.embedding_history[-50:]


@dataclass
class GlobalTrack:
    """Cross-camera track for a single person."""
    person_id: str
    local_tracks: Dict[str, Track] = field(default_factory=dict)  # camera_id -> Track
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    cameras_visited: Set[str] = field(default_factory=set)
    transition_history: List[Dict] = field(default_factory=list)
    is_known: bool = False
    known_person_id: Optional[str] = None
    known_person_name: Optional[str] = None
    
    @property
    def mean_embedding(self) -> Optional[np.ndarray]:
        """Get global mean embedding."""
        embeddings = []
        for track in self.local_tracks.values():
            if track.mean_embedding is not None:
                embeddings.append(track.mean_embedding)
        if embeddings:
            return np.mean(embeddings, axis=0)
        return None
    
    def add_transition(self, from_camera: str, to_camera: str, timestamp: datetime):
        """Record camera transition."""
        self.transition_history.append({
            "from": from_camera,
            "to": to_camera,
            "timestamp": timestamp.isoformat()
        })
    
    def get_timeline(self) -> List[Dict]:
        """Get timeline of appearances across cameras."""
        timeline = []
        
        for camera_id, track in self.local_tracks.items():
            for det in track.detections:
                timeline.append({
                    "timestamp": det.timestamp,
                    "camera_id": camera_id,
                    "confidence": det.confidence,
                    "snapshot": det.snapshot_path
                })
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])
        
        return timeline


class HungarianMatcher:
    """Hungarian algorithm for optimal matching."""
    
    @staticmethod
    def match(
        cost_matrix: np.ndarray,
        max_cost: float = 0.7
    ) -> List[Tuple[int, int]]:
        """
        Perform optimal matching using Hungarian algorithm.
        
        Args:
            cost_matrix: NxM cost matrix (lower is better)
            max_cost: Maximum cost for valid match
            
        Returns:
            List of (row_idx, col_idx) matches
        """
        from scipy.optimize import linear_sum_assignment
        
        if cost_matrix.size == 0:
            return []
        
        # Run Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filter by max cost
        matches = []
        for row_idx, col_idx in zip(row_indices, col_indices):
            if cost_matrix[row_idx, col_idx] <= max_cost:
                matches.append((row_idx, col_idx))
        
        return matches


class SingleCameraTracker:
    """Tracker for a single camera feed."""
    
    def __init__(
        self,
        camera_id: str,
        max_age: int = 30,  # Frames before track is lost
        min_hits: int = 3,  # Minimum detections to confirm track
        iou_threshold: float = 0.3,
        embedding_threshold: float = 0.6
    ):
        """
        Initialize single camera tracker.
        
        Args:
            camera_id: Camera identifier
            max_age: Maximum frames to keep lost track
            min_hits: Minimum hits to confirm track
            iou_threshold: IoU threshold for matching
            embedding_threshold: Embedding distance threshold
        """
        self.camera_id = camera_id
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.embedding_threshold = embedding_threshold
        
        self.tracks: Dict[str, Track] = {}
        self.frame_count = 0
        self._next_track_id = 0
    
    def update(
        self,
        detections: List[Detection],
        timestamp: datetime
    ) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections in current frame
            timestamp: Current timestamp
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Get active tracks
        active_tracks = [t for t in self.tracks.values() if t.state == "active"]
        
        if not active_tracks and not detections:
            return []
        
        if not active_tracks:
            # Create new tracks for all detections
            for det in detections:
                self._create_track(det)
            return list(self.tracks.values())
        
        if not detections:
            # Age all tracks
            self._age_tracks()
            return [t for t in self.tracks.values() if t.state == "active"]
        
        # Build cost matrix
        cost_matrix = self._build_cost_matrix(active_tracks, detections)
        
        # Match using Hungarian algorithm
        matches = HungarianMatcher.match(cost_matrix, max_cost=self.embedding_threshold)
        
        # Update matched tracks
        matched_track_indices = set()
        matched_det_indices = set()
        
        for track_idx, det_idx in matches:
            track = active_tracks[track_idx]
            detection = detections[det_idx]
            track.add_detection(detection)
            matched_track_indices.add(track_idx)
            matched_det_indices.add(det_idx)
        
        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_det_indices:
                self._create_track(det)
        
        # Age unmatched tracks
        for track_idx, track in enumerate(active_tracks):
            if track_idx not in matched_track_indices:
                frames_since_seen = self.frame_count - (
                    track.detections[-1].frame_number if track.detections else 0
                )
                if frames_since_seen > self.max_age:
                    track.state = "lost"
        
        return [t for t in self.tracks.values() if t.state == "active"]
    
    def _build_cost_matrix(
        self,
        tracks: List[Track],
        detections: List[Detection]
    ) -> np.ndarray:
        """Build cost matrix for matching."""
        n_tracks = len(tracks)
        n_dets = len(detections)
        cost_matrix = np.ones((n_tracks, n_dets))
        
        for i, track in enumerate(tracks):
            track_embedding = track.mean_embedding
            if track_embedding is None:
                continue
            
            for j, det in enumerate(detections):
                # Embedding distance (cosine)
                similarity = np.dot(track_embedding, det.embedding) / (
                    np.linalg.norm(track_embedding) * np.linalg.norm(det.embedding) + 1e-8
                )
                cost_matrix[i, j] = 1 - similarity
                
                # Optionally add IoU component
                if track.detections:
                    iou = self._calculate_iou(track.detections[-1].bbox, det.bbox)
                    cost_matrix[i, j] = 0.5 * cost_matrix[i, j] + 0.5 * (1 - iou)
        
        return cost_matrix
    
    def _calculate_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate IoU between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to corners
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xb - xa) * max(0, yb - ya)
        
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        iou = inter_area / (box1_area + box2_area - inter_area + 1e-8)
        
        return iou
    
    def _create_track(self, detection: Detection):
        """Create a new track."""
        track_id = f"{self.camera_id}_track_{self._next_track_id}"
        self._next_track_id += 1
        
        track = Track(
            track_id=track_id,
            camera_id=self.camera_id
        )
        track.add_detection(detection)
        
        self.tracks[track_id] = track
    
    def _age_tracks(self):
        """Age all tracks that weren't updated."""
        for track in self.tracks.values():
            if track.state == "active":
                if track.detections:
                    frames_since_seen = self.frame_count - track.detections[-1].frame_number
                    if frames_since_seen > self.max_age:
                        track.state = "lost"
    
    def get_confirmed_tracks(self) -> List[Track]:
        """Get tracks that have been confirmed (enough detections)."""
        return [
            t for t in self.tracks.values()
            if t.state == "active" and len(t.detections) >= self.min_hits
        ]


class MultiCameraTracker:
    """
    Multi-camera person tracking with re-identification.
    
    Features:
    - Cross-camera person re-identification
    - Global track management
    - Transition detection and logging
    - Timeline generation
    """
    
    def __init__(
        self,
        reid_threshold: float = 0.5,
        gallery_size: int = 100,
        transition_timeout: float = 300.0,  # 5 minutes
        known_faces_callback: Optional[callable] = None
    ):
        """
        Initialize multi-camera tracker.
        
        Args:
            reid_threshold: Threshold for re-identification matching
            gallery_size: Size of embedding gallery per global track
            transition_timeout: Max time between cameras for same person
            known_faces_callback: Callback to check if embedding matches known face
        """
        self.reid_threshold = reid_threshold
        self.gallery_size = gallery_size
        self.transition_timeout = transition_timeout
        self.known_faces_callback = known_faces_callback
        
        # Per-camera trackers
        self.camera_trackers: Dict[str, SingleCameraTracker] = {}
        
        # Global tracks
        self.global_tracks: Dict[str, GlobalTrack] = {}
        
        # Mapping from local track to global track
        self.local_to_global: Dict[str, str] = {}
        
        self._lock = threading.Lock()
    
    def add_camera(self, camera_id: str, **kwargs):
        """Add a camera to the tracking system."""
        with self._lock:
            if camera_id not in self.camera_trackers:
                self.camera_trackers[camera_id] = SingleCameraTracker(
                    camera_id=camera_id,
                    **kwargs
                )
                logger.info(f"Added camera: {camera_id}")
    
    def remove_camera(self, camera_id: str):
        """Remove a camera from the tracking system."""
        with self._lock:
            if camera_id in self.camera_trackers:
                del self.camera_trackers[camera_id]
                logger.info(f"Removed camera: {camera_id}")
    
    def update(
        self,
        camera_id: str,
        detections: List[Detection],
        timestamp: datetime
    ) -> Dict:
        """
        Update tracking for a camera.
        
        Args:
            camera_id: Camera identifier
            detections: List of detections
            timestamp: Current timestamp
            
        Returns:
            Update results with track info
        """
        with self._lock:
            # Ensure camera exists
            if camera_id not in self.camera_trackers:
                self.add_camera(camera_id)
            
            # Update local tracker
            tracker = self.camera_trackers[camera_id]
            active_tracks = tracker.update(detections, timestamp)
            
            # Process confirmed tracks for re-ID
            confirmed_tracks = tracker.get_confirmed_tracks()
            
            results = {
                "camera_id": camera_id,
                "active_tracks": len(active_tracks),
                "confirmed_tracks": len(confirmed_tracks),
                "new_global_tracks": 0,
                "reid_matches": 0,
                "transitions": []
            }
            
            for track in confirmed_tracks:
                if track.track_id in self.local_to_global:
                    # Already assigned to global track
                    global_id = self.local_to_global[track.track_id]
                    global_track = self.global_tracks[global_id]
                    global_track.last_seen = timestamp
                    global_track.local_tracks[camera_id] = track
                else:
                    # Try to match to existing global track
                    match = self._find_global_match(track, timestamp)
                    
                    if match:
                        # Re-identification success
                        global_id = match
                        global_track = self.global_tracks[global_id]
                        
                        # Check for transition
                        if camera_id not in global_track.cameras_visited:
                            last_camera = list(global_track.local_tracks.keys())[-1] if global_track.local_tracks else None
                            if last_camera:
                                global_track.add_transition(last_camera, camera_id, timestamp)
                                results["transitions"].append({
                                    "person_id": global_id,
                                    "from": last_camera,
                                    "to": camera_id
                                })
                        
                        global_track.cameras_visited.add(camera_id)
                        global_track.local_tracks[camera_id] = track
                        global_track.last_seen = timestamp
                        
                        self.local_to_global[track.track_id] = global_id
                        track.person_id = global_id
                        
                        results["reid_matches"] += 1
                    else:
                        # Create new global track
                        global_id = self._create_global_track(track, camera_id, timestamp)
                        results["new_global_tracks"] += 1
            
            return results
    
    def _find_global_match(
        self,
        track: Track,
        timestamp: datetime
    ) -> Optional[str]:
        """Find matching global track for re-identification."""
        track_embedding = track.mean_embedding
        if track_embedding is None:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for global_id, global_track in self.global_tracks.items():
            # Skip if same camera (should use local tracking)
            if track.camera_id in global_track.local_tracks:
                continue
            
            # Check timeout
            if global_track.last_seen:
                time_diff = (timestamp - global_track.last_seen).total_seconds()
                if time_diff > self.transition_timeout:
                    continue
            
            global_embedding = global_track.mean_embedding
            if global_embedding is None:
                continue
            
            # Calculate similarity
            similarity = np.dot(track_embedding, global_embedding) / (
                np.linalg.norm(track_embedding) * np.linalg.norm(global_embedding) + 1e-8
            )
            
            if similarity > best_similarity and similarity > (1 - self.reid_threshold):
                best_similarity = similarity
                best_match = global_id
        
        return best_match
    
    def _create_global_track(
        self,
        track: Track,
        camera_id: str,
        timestamp: datetime
    ) -> str:
        """Create a new global track."""
        global_id = f"person_{uuid.uuid4().hex[:8]}"
        
        global_track = GlobalTrack(
            person_id=global_id,
            first_seen=timestamp,
            last_seen=timestamp
        )
        global_track.local_tracks[camera_id] = track
        global_track.cameras_visited.add(camera_id)
        
        # Check against known faces
        if self.known_faces_callback and track.mean_embedding is not None:
            known_match = self.known_faces_callback(track.mean_embedding)
            if known_match:
                global_track.is_known = True
                global_track.known_person_id = known_match.get('person_id')
                global_track.known_person_name = known_match.get('name')
        
        self.global_tracks[global_id] = global_track
        self.local_to_global[track.track_id] = global_id
        track.person_id = global_id
        
        return global_id
    
    def get_global_track(self, person_id: str) -> Optional[GlobalTrack]:
        """Get a global track by person ID."""
        return self.global_tracks.get(person_id)
    
    def get_all_global_tracks(
        self,
        active_only: bool = True,
        min_cameras: int = 1
    ) -> List[GlobalTrack]:
        """Get all global tracks."""
        tracks = list(self.global_tracks.values())
        
        if min_cameras > 1:
            tracks = [t for t in tracks if len(t.cameras_visited) >= min_cameras]
        
        # Sort by last seen
        tracks.sort(key=lambda t: t.last_seen or datetime.min, reverse=True)
        
        return tracks
    
    def get_person_timeline(self, person_id: str) -> List[Dict]:
        """Get timeline for a specific person."""
        global_track = self.global_tracks.get(person_id)
        if not global_track:
            return []
        return global_track.get_timeline()
    
    def get_camera_transitions(
        self,
        person_id: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[Dict]:
        """Get camera transition events."""
        transitions = []
        
        tracks = [self.global_tracks[person_id]] if person_id else self.global_tracks.values()
        
        for track in tracks:
            for transition in track.transition_history:
                if time_range:
                    ts = datetime.fromisoformat(transition['timestamp'])
                    if not (time_range[0] <= ts <= time_range[1]):
                        continue
                
                transitions.append({
                    "person_id": track.person_id,
                    "person_name": track.known_person_name,
                    **transition
                })
        
        # Sort by timestamp
        transitions.sort(key=lambda x: x['timestamp'])
        
        return transitions
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics."""
        total_local_tracks = sum(
            len(t.tracks) for t in self.camera_trackers.values()
        )
        
        active_local_tracks = sum(
            len([tr for tr in t.tracks.values() if tr.state == "active"])
            for t in self.camera_trackers.values()
        )
        
        return {
            "cameras": len(self.camera_trackers),
            "global_tracks": len(self.global_tracks),
            "total_local_tracks": total_local_tracks,
            "active_local_tracks": active_local_tracks,
            "known_persons": sum(1 for t in self.global_tracks.values() if t.is_known),
            "multi_camera_tracks": sum(
                1 for t in self.global_tracks.values() 
                if len(t.cameras_visited) > 1
            )
        }
    
    def visualize_movement_graph(self) -> Dict:
        """
        Generate movement graph data for visualization.
        
        Returns:
            Graph data with nodes (cameras) and edges (transitions)
        """
        nodes = [{"id": cam_id, "label": cam_id} for cam_id in self.camera_trackers]
        
        # Count transitions between cameras
        edge_counts = defaultdict(int)
        for track in self.global_tracks.values():
            for transition in track.transition_history:
                edge_key = (transition['from'], transition['to'])
                edge_counts[edge_key] += 1
        
        edges = [
            {
                "from": from_cam,
                "to": to_cam,
                "weight": count,
                "label": str(count)
            }
            for (from_cam, to_cam), count in edge_counts.items()
        ]
        
        return {"nodes": nodes, "edges": edges}
