"""
Face clustering module for grouping unknown persons.
Uses DBSCAN/HDBSCAN to cluster face embeddings and track recurring unknowns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class UnknownFace:
    """Represents an unknown face detection."""
    face_id: str
    embedding: np.ndarray
    timestamp: datetime
    camera_id: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    snapshot_path: Optional[str] = None
    cluster_id: Optional[int] = None


@dataclass
class FaceCluster:
    """Represents a cluster of similar unknown faces."""
    cluster_id: int
    faces: List[UnknownFace] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    camera_appearances: Dict[str, int] = field(default_factory=dict)
    is_person_of_interest: bool = False
    alert_triggered: bool = False
    notes: str = ""
    
    def update_stats(self):
        """Update cluster statistics."""
        if self.faces:
            self.first_seen = min(f.timestamp for f in self.faces)
            self.last_seen = max(f.timestamp for f in self.faces)
            
            # Update camera appearances
            self.camera_appearances = defaultdict(int)
            for face in self.faces:
                self.camera_appearances[face.camera_id] += 1
            
            # Update centroid
            embeddings = np.array([f.embedding for f in self.faces])
            self.centroid = np.mean(embeddings, axis=0)
    
    @property
    def appearance_count(self) -> int:
        return len(self.faces)
    
    @property
    def unique_cameras(self) -> int:
        return len(self.camera_appearances)


class FaceClusteringEngine:
    """
    Engine for clustering unknown faces using DBSCAN/HDBSCAN.
    
    Features:
    - Real-time clustering of face embeddings
    - Person of interest tracking
    - Frequency-based alerts
    - Cross-camera appearance tracking
    """
    
    def __init__(
        self,
        algorithm: str = "hdbscan",  # "dbscan" or "hdbscan"
        min_cluster_size: int = 3,
        min_samples: int = 2,
        eps: float = 0.5,  # For DBSCAN
        distance_threshold: float = 0.6,  # For matching to existing clusters
        alert_threshold: int = 5,  # Appearances before alert
        poi_threshold: int = 10,  # Appearances to mark as person of interest
        max_faces_memory: int = 10000,
        persistence_path: Optional[str] = None
    ):
        """
        Initialize face clustering engine.
        
        Args:
            algorithm: Clustering algorithm ("dbscan" or "hdbscan")
            min_cluster_size: Minimum faces to form a cluster
            min_samples: Min samples for core point (DBSCAN)
            eps: Maximum distance for DBSCAN
            distance_threshold: Max distance to assign to existing cluster
            alert_threshold: Appearances before triggering alert
            poi_threshold: Appearances to mark as person of interest
            max_faces_memory: Maximum faces to keep in memory
            persistence_path: Path to save/load cluster data
        """
        self.algorithm = algorithm
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.eps = eps
        self.distance_threshold = distance_threshold
        self.alert_threshold = alert_threshold
        self.poi_threshold = poi_threshold
        self.max_faces_memory = max_faces_memory
        self.persistence_path = Path(persistence_path) if persistence_path else None
        
        # Storage
        self.unknown_faces: List[UnknownFace] = []
        self.clusters: Dict[int, FaceCluster] = {}
        self.next_cluster_id = 0
        
        # Tracking
        self.poi_set: Set[int] = set()
        self.alert_callbacks: List[callable] = []
        
        self._lock = threading.Lock()
        self._clusterer = None
        
        # Load persisted data
        if self.persistence_path and self.persistence_path.exists():
            self._load_state()
    
    def _get_clusterer(self):
        """Get or create the clustering model."""
        if self._clusterer is None:
            if self.algorithm == "hdbscan":
                try:
                    import hdbscan
                    self._clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=self.min_cluster_size,
                        min_samples=self.min_samples,
                        metric='euclidean',
                        cluster_selection_method='eom'
                    )
                except ImportError:
                    logger.warning("HDBSCAN not available, falling back to DBSCAN")
                    self.algorithm = "dbscan"
            
            if self.algorithm == "dbscan":
                from sklearn.cluster import DBSCAN
                self._clusterer = DBSCAN(
                    eps=self.eps,
                    min_samples=self.min_samples,
                    metric='euclidean'
                )
        
        return self._clusterer
    
    def add_unknown_face(
        self,
        embedding: np.ndarray,
        camera_id: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        snapshot_path: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Add an unknown face and attempt to cluster it.
        
        Args:
            embedding: Face embedding vector
            camera_id: Camera where face was detected
            confidence: Detection confidence
            bbox: Bounding box (x, y, w, h)
            snapshot_path: Path to saved face image
            timestamp: Detection timestamp
            
        Returns:
            Result with cluster assignment and any alerts
        """
        with self._lock:
            # Create face record
            face = UnknownFace(
                face_id=f"face_{len(self.unknown_faces)}_{datetime.now().timestamp()}",
                embedding=embedding,
                timestamp=timestamp or datetime.now(),
                camera_id=camera_id,
                confidence=confidence,
                bbox=bbox,
                snapshot_path=snapshot_path
            )
            
            # Try to match to existing cluster
            matched_cluster = self._find_matching_cluster(embedding)
            
            if matched_cluster is not None:
                # Add to existing cluster
                face.cluster_id = matched_cluster
                self.clusters[matched_cluster].faces.append(face)
                self.clusters[matched_cluster].update_stats()
                
                result = self._check_cluster_alerts(matched_cluster)
            else:
                # Add to unknown pool for later clustering
                face.cluster_id = None
                result = {"cluster_id": None, "status": "pending_clustering"}
            
            self.unknown_faces.append(face)
            
            # Prune old faces if needed
            if len(self.unknown_faces) > self.max_faces_memory:
                self._prune_old_faces()
            
            return result
    
    def _find_matching_cluster(self, embedding: np.ndarray) -> Optional[int]:
        """Find a matching cluster for the embedding."""
        if not self.clusters:
            return None
        
        best_match = None
        best_distance = float('inf')
        
        for cluster_id, cluster in self.clusters.items():
            if cluster.centroid is not None:
                distance = np.linalg.norm(embedding - cluster.centroid)
                if distance < best_distance and distance < self.distance_threshold:
                    best_distance = distance
                    best_match = cluster_id
        
        return best_match
    
    def run_clustering(self, force: bool = False) -> Dict:
        """
        Run clustering on unclustered faces.
        
        Args:
            force: Force reclustering of all faces
            
        Returns:
            Clustering results
        """
        with self._lock:
            # Get unclustered faces
            if force:
                faces_to_cluster = self.unknown_faces
            else:
                faces_to_cluster = [f for f in self.unknown_faces if f.cluster_id is None]
            
            if len(faces_to_cluster) < self.min_cluster_size:
                return {
                    "status": "insufficient_data",
                    "faces_pending": len(faces_to_cluster),
                    "min_required": self.min_cluster_size
                }
            
            # Extract embeddings
            embeddings = np.array([f.embedding for f in faces_to_cluster])
            
            # Run clustering
            clusterer = self._get_clusterer()
            labels = clusterer.fit_predict(embeddings)
            
            # Process results
            new_clusters = 0
            updated_clusters = 0
            
            for face, label in zip(faces_to_cluster, labels):
                if label == -1:
                    # Noise point
                    continue
                
                # Map label to our cluster ID
                if label not in self._label_to_cluster:
                    # Create new cluster
                    cluster_id = self.next_cluster_id
                    self.next_cluster_id += 1
                    self.clusters[cluster_id] = FaceCluster(cluster_id=cluster_id)
                    self._label_to_cluster[label] = cluster_id
                    new_clusters += 1
                else:
                    cluster_id = self._label_to_cluster[label]
                    updated_clusters += 1
                
                face.cluster_id = cluster_id
                
                # Add to cluster if not already there
                if face not in self.clusters[cluster_id].faces:
                    self.clusters[cluster_id].faces.append(face)
            
            # Update all cluster stats
            for cluster in self.clusters.values():
                cluster.update_stats()
            
            # Check for alerts
            alerts = []
            for cluster_id in self.clusters:
                alert = self._check_cluster_alerts(cluster_id)
                if alert.get('alert'):
                    alerts.append(alert)
            
            return {
                "status": "completed",
                "new_clusters": new_clusters,
                "updated_clusters": updated_clusters,
                "total_clusters": len(self.clusters),
                "alerts": alerts
            }
    
    @property
    def _label_to_cluster(self) -> Dict:
        """Mapping from clustering labels to our cluster IDs."""
        if not hasattr(self, '_label_map'):
            self._label_map = {}
        return self._label_map
    
    def _check_cluster_alerts(self, cluster_id: int) -> Dict:
        """Check if cluster triggers any alerts."""
        cluster = self.clusters.get(cluster_id)
        if not cluster:
            return {}
        
        result = {
            "cluster_id": cluster_id,
            "appearance_count": cluster.appearance_count,
            "unique_cameras": cluster.unique_cameras
        }
        
        # Check for person of interest
        if (cluster.appearance_count >= self.poi_threshold and 
            cluster_id not in self.poi_set):
            cluster.is_person_of_interest = True
            self.poi_set.add(cluster_id)
            result["new_poi"] = True
        
        # Check for alert threshold
        if (cluster.appearance_count >= self.alert_threshold and 
            not cluster.alert_triggered):
            cluster.alert_triggered = True
            result["alert"] = True
            result["alert_type"] = "frequency_threshold"
            result["message"] = f"Unknown person seen {cluster.appearance_count} times across {cluster.unique_cameras} cameras"
            
            # Trigger callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
        
        return result
    
    def get_cluster(self, cluster_id: int) -> Optional[FaceCluster]:
        """Get a specific cluster."""
        return self.clusters.get(cluster_id)
    
    def get_all_clusters(
        self,
        min_appearances: int = 1,
        poi_only: bool = False
    ) -> List[FaceCluster]:
        """Get all clusters matching criteria."""
        clusters = list(self.clusters.values())
        
        if min_appearances > 1:
            clusters = [c for c in clusters if c.appearance_count >= min_appearances]
        
        if poi_only:
            clusters = [c for c in clusters if c.is_person_of_interest]
        
        # Sort by appearance count
        clusters.sort(key=lambda c: c.appearance_count, reverse=True)
        
        return clusters
    
    def get_cluster_timeline(
        self,
        cluster_id: int
    ) -> List[Dict]:
        """Get timeline of appearances for a cluster."""
        cluster = self.clusters.get(cluster_id)
        if not cluster:
            return []
        
        timeline = []
        for face in sorted(cluster.faces, key=lambda f: f.timestamp):
            timeline.append({
                "timestamp": face.timestamp.isoformat(),
                "camera_id": face.camera_id,
                "confidence": face.confidence,
                "snapshot": face.snapshot_path
            })
        
        return timeline
    
    def find_similar_clusters(
        self,
        embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Find clusters most similar to a given embedding."""
        if not self.clusters:
            return []
        
        similarities = []
        for cluster_id, cluster in self.clusters.items():
            if cluster.centroid is not None:
                distance = np.linalg.norm(embedding - cluster.centroid)
                similarity = 1.0 / (1.0 + distance)
                similarities.append((cluster_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def merge_clusters(self, cluster_ids: List[int]) -> Optional[int]:
        """Merge multiple clusters into one."""
        with self._lock:
            if len(cluster_ids) < 2:
                return None
            
            # Keep the first cluster as the target
            target_id = cluster_ids[0]
            target = self.clusters.get(target_id)
            
            if not target:
                return None
            
            # Merge others into target
            for cluster_id in cluster_ids[1:]:
                source = self.clusters.get(cluster_id)
                if source:
                    target.faces.extend(source.faces)
                    for face in source.faces:
                        face.cluster_id = target_id
                    del self.clusters[cluster_id]
                    self.poi_set.discard(cluster_id)
            
            target.update_stats()
            return target_id
    
    def mark_as_known(
        self,
        cluster_id: int,
        person_id: str,
        person_name: str
    ) -> bool:
        """Mark a cluster as a known person (moves to known database)."""
        cluster = self.clusters.get(cluster_id)
        if not cluster:
            return False
        
        # Return the embedding for the known person database
        # The caller should add this to the known faces database
        logger.info(f"Cluster {cluster_id} identified as {person_name} (ID: {person_id})")
        
        # Remove from unknown clusters
        with self._lock:
            del self.clusters[cluster_id]
            self.poi_set.discard(cluster_id)
            
            # Update faces
            for face in self.unknown_faces:
                if face.cluster_id == cluster_id:
                    face.cluster_id = None  # Mark as resolved
        
        return True
    
    def register_alert_callback(self, callback: callable):
        """Register a callback for cluster alerts."""
        self.alert_callbacks.append(callback)
    
    def _prune_old_faces(self, max_age_hours: int = 168):  # 1 week default
        """Remove old faces to free memory."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        self.unknown_faces = [
            f for f in self.unknown_faces
            if f.timestamp > cutoff
        ]
        
        # Update clusters
        for cluster in self.clusters.values():
            cluster.faces = [f for f in cluster.faces if f.timestamp > cutoff]
            cluster.update_stats()
        
        # Remove empty clusters
        empty_clusters = [cid for cid, c in self.clusters.items() if not c.faces]
        for cid in empty_clusters:
            del self.clusters[cid]
            self.poi_set.discard(cid)
    
    def _save_state(self):
        """Save state to disk."""
        if not self.persistence_path:
            return
        
        try:
            state = {
                'unknown_faces': self.unknown_faces,
                'clusters': self.clusters,
                'next_cluster_id': self.next_cluster_id,
                'poi_set': self.poi_set
            }
            
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persistence_path, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"Saved clustering state to {self.persistence_path}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _load_state(self):
        """Load state from disk."""
        try:
            with open(self.persistence_path, 'rb') as f:
                state = pickle.load(f)
            
            self.unknown_faces = state.get('unknown_faces', [])
            self.clusters = state.get('clusters', {})
            self.next_cluster_id = state.get('next_cluster_id', 0)
            self.poi_set = state.get('poi_set', set())
            
            logger.info(f"Loaded clustering state: {len(self.clusters)} clusters")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    def get_statistics(self) -> Dict:
        """Get clustering statistics."""
        return {
            "total_unknown_faces": len(self.unknown_faces),
            "total_clusters": len(self.clusters),
            "persons_of_interest": len(self.poi_set),
            "unclustered_faces": sum(1 for f in self.unknown_faces if f.cluster_id is None),
            "largest_cluster": max(
                (c.appearance_count for c in self.clusters.values()),
                default=0
            ),
            "avg_cluster_size": (
                sum(c.appearance_count for c in self.clusters.values()) / len(self.clusters)
                if self.clusters else 0
            )
        }
