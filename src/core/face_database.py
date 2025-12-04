"""
Searchable Face Database Module

Advanced face search with:
- Similarity-based face search
- Time-range queries
- Clip/snapshot export
- Multi-attribute filtering
- Elasticsearch/FAISS integration

Author: CCTV Face Detection System
"""

import asyncio
import io
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib
import json
import struct
import zipfile

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Try FAISS import
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Using numpy-based similarity search.")


class SearchType(Enum):
    """Types of face searches"""
    SIMILARITY = "similarity"  # Find similar faces by embedding
    EXACT_MATCH = "exact_match"  # Find specific person
    TIME_RANGE = "time_range"  # Search within time period
    CAMERA = "camera"  # Search by camera
    COMBINED = "combined"  # Multi-filter search


@dataclass
class FaceRecord:
    """A single face detection record"""
    record_id: str
    person_id: Optional[str]  # None if unknown
    person_name: Optional[str]
    embedding: np.ndarray
    confidence: float
    camera_id: str
    camera_name: str
    timestamp: datetime
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    snapshot_path: Optional[str] = None
    video_clip_path: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)  # age, gender, emotion, etc.
    is_known: bool = False
    liveness_score: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "record_id": self.record_id,
            "person_id": self.person_id,
            "person_name": self.person_name,
            "confidence": self.confidence,
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "timestamp": self.timestamp.isoformat(),
            "bounding_box": self.bounding_box,
            "snapshot_path": self.snapshot_path,
            "video_clip_path": self.video_clip_path,
            "attributes": self.attributes,
            "is_known": self.is_known,
            "liveness_score": self.liveness_score
        }


@dataclass
class SearchQuery:
    """Query parameters for face search"""
    # Embedding search
    query_embedding: Optional[np.ndarray] = None
    similarity_threshold: float = 0.6
    
    # Person search
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    
    # Time filters
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Location filters
    camera_ids: Optional[List[str]] = None
    
    # Attribute filters
    min_confidence: float = 0.0
    is_known: Optional[bool] = None  # None = both, True/False = filter
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    gender: Optional[str] = None
    emotion: Optional[str] = None
    
    # Pagination
    limit: int = 100
    offset: int = 0
    
    # Sort
    sort_by: str = "timestamp"  # 'timestamp', 'confidence', 'similarity'
    sort_desc: bool = True


@dataclass
class SearchResult:
    """Result of a face search"""
    record: FaceRecord
    similarity_score: Optional[float] = None
    rank: int = 0


class FAISSIndex:
    """
    FAISS-based vector similarity index for fast face search.
    
    Uses IVF (Inverted File) index with PQ (Product Quantization)
    for efficient billion-scale search.
    """
    
    def __init__(
        self,
        dimension: int = 512,
        index_type: str = "IVF_PQ",
        nlist: int = 100,  # Number of clusters
        m: int = 8  # Number of sub-quantizers
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.m = m
        self.index = None
        self.id_map: Dict[int, str] = {}  # FAISS id -> record_id
        self.is_trained = False
        
    def build_index(self, embeddings: np.ndarray, record_ids: List[str]):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: (N, D) array of face embeddings
            record_ids: Corresponding record IDs
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not installed. pip install faiss-cpu")
            
        n_samples = len(embeddings)
        
        if self.index_type == "Flat":
            # Exact search (slow but accurate)
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IVF_Flat":
            # Inverted file with exact vectors
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
        elif self.index_type == "IVF_PQ":
            # Inverted file with product quantization (most efficient)
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFPQ(
                quantizer, self.dimension, self.nlist, self.m, 8
            )
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World graphs
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Train index if needed
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            if n_samples < self.nlist:
                # Not enough samples, use flat index
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.index.train(embeddings)
        
        # Add vectors
        self.index.add(embeddings)
        
        # Store ID mapping
        self.id_map = {i: rid for i, rid in enumerate(record_ids)}
        self.is_trained = True
        
        logger.info(f"Built FAISS index with {n_samples} vectors")
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search for similar faces.
        
        Args:
            query: Query embedding (D,) or (N, D)
            k: Number of results
            
        Returns:
            List of (record_id, similarity_score)
        """
        if self.index is None or not self.is_trained:
            return []
            
        # Ensure proper shape
        if query.ndim == 1:
            query = query.reshape(1, -1)
        query = query.astype(np.float32)
        faiss.normalize_L2(query)
        
        # Search
        scores, indices = self.index.search(query, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx in self.id_map:
                results.append((self.id_map[idx], float(score)))
                
        return results
    
    def add_vectors(self, embeddings: np.ndarray, record_ids: List[str]):
        """Add new vectors to existing index"""
        if self.index is None:
            self.build_index(embeddings, record_ids)
            return
            
        start_id = len(self.id_map)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        for i, rid in enumerate(record_ids):
            self.id_map[start_id + i] = rid
    
    def save(self, path: str):
        """Save index to disk"""
        if self.index is not None:
            faiss.write_index(self.index, f"{path}.faiss")
            with open(f"{path}.map", 'wb') as f:
                pickle.dump(self.id_map, f)
    
    def load(self, path: str):
        """Load index from disk"""
        if Path(f"{path}.faiss").exists():
            self.index = faiss.read_index(f"{path}.faiss")
            with open(f"{path}.map", 'rb') as f:
                self.id_map = pickle.load(f)
            self.is_trained = True


class NumpyFallbackIndex:
    """
    Numpy-based similarity search when FAISS is not available.
    
    Slower but works without additional dependencies.
    """
    
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.embeddings: Optional[np.ndarray] = None
        self.record_ids: List[str] = []
        
    def build_index(self, embeddings: np.ndarray, record_ids: List[str]):
        """Build index (just stores normalized embeddings)"""
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = embeddings / (norms + 1e-10)
        self.record_ids = list(record_ids)
        
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search using numpy dot product"""
        if self.embeddings is None or len(self.record_ids) == 0:
            return []
            
        # Normalize query
        query = query.flatten()
        query = query / (np.linalg.norm(query) + 1e-10)
        
        # Compute cosine similarities
        similarities = self.embeddings @ query
        
        # Get top-k
        top_k_idx = np.argsort(similarities)[::-1][:k]
        
        return [
            (self.record_ids[idx], float(similarities[idx]))
            for idx in top_k_idx
        ]
    
    def add_vectors(self, embeddings: np.ndarray, record_ids: List[str]):
        """Add new vectors"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)
        
        if self.embeddings is None:
            self.embeddings = normalized
        else:
            self.embeddings = np.vstack([self.embeddings, normalized])
        self.record_ids.extend(record_ids)
    
    def save(self, path: str):
        """Save to disk"""
        np.savez(
            f"{path}.npz",
            embeddings=self.embeddings,
            record_ids=np.array(self.record_ids, dtype=object)
        )
    
    def load(self, path: str):
        """Load from disk"""
        data = np.load(f"{path}.npz", allow_pickle=True)
        self.embeddings = data['embeddings']
        self.record_ids = list(data['record_ids'])


class FaceDatabase:
    """
    Main searchable face database.
    
    Combines vector similarity search with SQL-like filtering
    for comprehensive face search capabilities.
    """
    
    def __init__(
        self,
        storage_path: str = "data/face_db",
        embedding_dim: int = 512,
        use_faiss: bool = True
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.embedding_dim = embedding_dim
        
        # Initialize vector index
        if use_faiss and FAISS_AVAILABLE:
            self.index = FAISSIndex(dimension=embedding_dim)
        else:
            self.index = NumpyFallbackIndex(dimension=embedding_dim)
        
        # In-memory record storage (could use SQLite for persistence)
        self.records: Dict[str, FaceRecord] = {}
        
        # Secondary indexes for fast filtering
        self.by_person: Dict[str, List[str]] = {}  # person_id -> record_ids
        self.by_camera: Dict[str, List[str]] = {}  # camera_id -> record_ids
        self.by_date: Dict[str, List[str]] = {}  # 'YYYY-MM-DD' -> record_ids
        
        # Load existing data
        self._load_records()
    
    def _load_records(self):
        """Load records from disk"""
        records_file = self.storage_path / "records.pkl"
        if records_file.exists():
            with open(records_file, 'rb') as f:
                self.records = pickle.load(f)
                
            # Rebuild indexes
            self._rebuild_indexes()
            logger.info(f"Loaded {len(self.records)} face records")
    
    def _save_records(self):
        """Save records to disk"""
        with open(self.storage_path / "records.pkl", 'wb') as f:
            pickle.dump(self.records, f)
    
    def _rebuild_indexes(self):
        """Rebuild secondary indexes from records"""
        self.by_person.clear()
        self.by_camera.clear()
        self.by_date.clear()
        
        embeddings = []
        record_ids = []
        
        for rid, record in self.records.items():
            # Person index
            if record.person_id:
                if record.person_id not in self.by_person:
                    self.by_person[record.person_id] = []
                self.by_person[record.person_id].append(rid)
            
            # Camera index
            if record.camera_id not in self.by_camera:
                self.by_camera[record.camera_id] = []
            self.by_camera[record.camera_id].append(rid)
            
            # Date index
            date_key = record.timestamp.strftime("%Y-%m-%d")
            if date_key not in self.by_date:
                self.by_date[date_key] = []
            self.by_date[date_key].append(rid)
            
            # Collect embeddings for vector index
            embeddings.append(record.embedding)
            record_ids.append(rid)
        
        # Build vector index
        if embeddings:
            embeddings = np.array(embeddings, dtype=np.float32)
            self.index.build_index(embeddings, record_ids)
    
    def add_record(self, record: FaceRecord):
        """
        Add a new face record to the database.
        
        Args:
            record: FaceRecord to add
        """
        self.records[record.record_id] = record
        
        # Update secondary indexes
        if record.person_id:
            if record.person_id not in self.by_person:
                self.by_person[record.person_id] = []
            self.by_person[record.person_id].append(record.record_id)
        
        if record.camera_id not in self.by_camera:
            self.by_camera[record.camera_id] = []
        self.by_camera[record.camera_id].append(record.record_id)
        
        date_key = record.timestamp.strftime("%Y-%m-%d")
        if date_key not in self.by_date:
            self.by_date[date_key] = []
        self.by_date[date_key].append(record.record_id)
        
        # Add to vector index
        embedding = record.embedding.reshape(1, -1).astype(np.float32)
        self.index.add_vectors(embedding, [record.record_id])
    
    def add_records_batch(self, records: List[FaceRecord]):
        """Add multiple records efficiently"""
        embeddings = []
        record_ids = []
        
        for record in records:
            self.records[record.record_id] = record
            
            if record.person_id:
                if record.person_id not in self.by_person:
                    self.by_person[record.person_id] = []
                self.by_person[record.person_id].append(record.record_id)
            
            if record.camera_id not in self.by_camera:
                self.by_camera[record.camera_id] = []
            self.by_camera[record.camera_id].append(record.record_id)
            
            date_key = record.timestamp.strftime("%Y-%m-%d")
            if date_key not in self.by_date:
                self.by_date[date_key] = []
            self.by_date[date_key].append(record.record_id)
            
            embeddings.append(record.embedding)
            record_ids.append(record.record_id)
        
        if embeddings:
            embeddings = np.array(embeddings, dtype=np.float32)
            self.index.add_vectors(embeddings, record_ids)
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Search the face database.
        
        Supports:
        - Similarity search by face embedding
        - Exact person match
        - Time range filtering
        - Camera filtering
        - Attribute filtering
        - Combined filters
        
        Args:
            query: SearchQuery with search parameters
            
        Returns:
            List of SearchResult sorted by relevance
        """
        candidate_ids: Optional[set] = None
        
        # ===== Apply filters to narrow candidates =====
        
        # Filter by person
        if query.person_id:
            person_records = set(self.by_person.get(query.person_id, []))
            candidate_ids = person_records if candidate_ids is None else candidate_ids & person_records
        
        # Filter by camera
        if query.camera_ids:
            camera_records = set()
            for cam_id in query.camera_ids:
                camera_records.update(self.by_camera.get(cam_id, []))
            candidate_ids = camera_records if candidate_ids is None else candidate_ids & camera_records
        
        # Filter by date range
        if query.start_time or query.end_time:
            date_records = set()
            
            start = query.start_time or datetime.min
            end = query.end_time or datetime.max
            
            for date_str, rids in self.by_date.items():
                date = datetime.strptime(date_str, "%Y-%m-%d")
                if start.date() <= date.date() <= end.date():
                    date_records.update(rids)
            
            candidate_ids = date_records if candidate_ids is None else candidate_ids & date_records
        
        # ===== Similarity search =====
        similarity_scores: Dict[str, float] = {}
        
        if query.query_embedding is not None:
            # Use vector index for similarity search
            k = min(len(self.records), 1000)  # Search top-1000 then filter
            similar = self.index.search(query.query_embedding, k=k)
            
            for rid, score in similar:
                if score >= query.similarity_threshold:
                    similarity_scores[rid] = score
            
            # Intersect with other filters
            if candidate_ids is not None:
                similarity_scores = {
                    k: v for k, v in similarity_scores.items() 
                    if k in candidate_ids
                }
            else:
                candidate_ids = set(similarity_scores.keys())
        
        # If no filters applied, use all records
        if candidate_ids is None:
            candidate_ids = set(self.records.keys())
        
        # ===== Apply remaining filters and build results =====
        results: List[SearchResult] = []
        
        for rid in candidate_ids:
            record = self.records.get(rid)
            if not record:
                continue
            
            # Filter by time (exact, not just date)
            if query.start_time and record.timestamp < query.start_time:
                continue
            if query.end_time and record.timestamp > query.end_time:
                continue
            
            # Filter by confidence
            if record.confidence < query.min_confidence:
                continue
            
            # Filter by is_known
            if query.is_known is not None and record.is_known != query.is_known:
                continue
            
            # Filter by name (partial match)
            if query.person_name:
                if not record.person_name or query.person_name.lower() not in record.person_name.lower():
                    continue
            
            # Filter by attributes
            attrs = record.attributes
            if query.min_age is not None and attrs.get('age', 0) < query.min_age:
                continue
            if query.max_age is not None and attrs.get('age', 100) > query.max_age:
                continue
            if query.gender and attrs.get('gender') != query.gender:
                continue
            if query.emotion and attrs.get('emotion') != query.emotion:
                continue
            
            # Add to results
            results.append(SearchResult(
                record=record,
                similarity_score=similarity_scores.get(rid),
                rank=0
            ))
        
        # ===== Sort results =====
        if query.sort_by == "similarity" and similarity_scores:
            results.sort(key=lambda r: r.similarity_score or 0, reverse=query.sort_desc)
        elif query.sort_by == "confidence":
            results.sort(key=lambda r: r.record.confidence, reverse=query.sort_desc)
        else:  # timestamp
            results.sort(key=lambda r: r.record.timestamp, reverse=query.sort_desc)
        
        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        # Apply pagination
        return results[query.offset:query.offset + query.limit]
    
    def search_by_photo(
        self,
        photo_embedding: np.ndarray,
        threshold: float = 0.6,
        limit: int = 50,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        camera_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Search by uploading a photo.
        
        Convenience method for similarity search.
        """
        query = SearchQuery(
            query_embedding=photo_embedding,
            similarity_threshold=threshold,
            limit=limit,
            start_time=time_range[0] if time_range else None,
            end_time=time_range[1] if time_range else None,
            camera_ids=camera_ids,
            sort_by="similarity"
        )
        return self.search(query)
    
    def get_person_timeline(
        self,
        person_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[FaceRecord]:
        """
        Get timeline of a person's appearances.
        
        Useful for tracking movement through different cameras.
        """
        query = SearchQuery(
            person_id=person_id,
            start_time=start_time,
            end_time=end_time,
            sort_by="timestamp",
            sort_desc=False,
            limit=1000
        )
        results = self.search(query)
        return [r.record for r in results]
    
    def get_camera_activity(
        self,
        camera_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get activity summary for a camera in time range"""
        query = SearchQuery(
            camera_ids=[camera_id],
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        results = self.search(query)
        
        known_count = sum(1 for r in results if r.record.is_known)
        unknown_count = len(results) - known_count
        unique_persons = len(set(r.record.person_id for r in results if r.record.person_id))
        
        return {
            "camera_id": camera_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_detections": len(results),
            "known_faces": known_count,
            "unknown_faces": unknown_count,
            "unique_persons": unique_persons,
            "records": [r.record.to_dict() for r in results[:100]]
        }
    
    def save(self):
        """Persist database to disk"""
        self._save_records()
        self.index.save(str(self.storage_path / "vector_index"))
        logger.info(f"Saved face database with {len(self.records)} records")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            "total_records": len(self.records),
            "unique_persons": len(self.by_person),
            "cameras": len(self.by_camera),
            "date_range": {
                "earliest": min(self.by_date.keys()) if self.by_date else None,
                "latest": max(self.by_date.keys()) if self.by_date else None
            },
            "storage_path": str(self.storage_path)
        }


class ClipExporter:
    """
    Export video clips and snapshots for search results.
    
    Generates:
    - Individual snapshots
    - Video clips around detection time
    - ZIP archives of multiple results
    """
    
    def __init__(
        self,
        video_storage_path: str,
        snapshot_storage_path: str,
        export_path: str = "exports"
    ):
        self.video_storage = Path(video_storage_path)
        self.snapshot_storage = Path(snapshot_storage_path)
        self.export_path = Path(export_path)
        self.export_path.mkdir(parents=True, exist_ok=True)
    
    def export_snapshot(
        self,
        record: FaceRecord,
        draw_bbox: bool = True,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Export snapshot for a face record.
        
        Args:
            record: Face record to export
            draw_bbox: Whether to draw bounding box
            output_path: Custom output path
            
        Returns:
            Path to exported snapshot
        """
        if not record.snapshot_path:
            return None
            
        src_path = self.snapshot_storage / record.snapshot_path
        if not src_path.exists():
            return None
            
        img = cv2.imread(str(src_path))
        if img is None:
            return None
        
        if draw_bbox:
            x, y, w, h = record.bounding_box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label
            label = record.person_name or "Unknown"
            cv2.putText(
                img, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
        
        if output_path is None:
            output_path = self.export_path / f"snapshot_{record.record_id}.jpg"
        
        cv2.imwrite(str(output_path), img)
        return str(output_path)
    
    def export_clip(
        self,
        record: FaceRecord,
        before_seconds: float = 5.0,
        after_seconds: float = 5.0,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Export video clip around detection time.
        
        Args:
            record: Face record
            before_seconds: Seconds before detection to include
            after_seconds: Seconds after detection to include
            output_path: Custom output path
            
        Returns:
            Path to exported clip
        """
        if not record.video_clip_path:
            return None
            
        src_video = self.video_storage / record.video_clip_path
        if not src_video.exists():
            return None
            
        if output_path is None:
            output_path = self.export_path / f"clip_{record.record_id}.mp4"
        
        # Open source video
        cap = cv2.VideoCapture(str(src_video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame range
        # This assumes video files are named/organized by timestamp
        # You'd need to adjust based on your video storage scheme
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ))
        
        total_frames = int((before_seconds + after_seconds) * fps)
        frames_written = 0
        
        while frames_written < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frames_written += 1
        
        cap.release()
        out.release()
        
        return str(output_path)
    
    def export_batch(
        self,
        records: List[FaceRecord],
        include_snapshots: bool = True,
        include_clips: bool = False,
        archive_name: Optional[str] = None
    ) -> str:
        """
        Export multiple records as a ZIP archive.
        
        Args:
            records: List of records to export
            include_snapshots: Include snapshot images
            include_clips: Include video clips
            archive_name: Custom archive name
            
        Returns:
            Path to ZIP archive
        """
        if archive_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"export_{timestamp}.zip"
        
        archive_path = self.export_path / archive_name
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add metadata JSON
            metadata = {
                "export_time": datetime.now().isoformat(),
                "record_count": len(records),
                "records": [r.to_dict() for r in records]
            }
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))
            
            # Add snapshots
            if include_snapshots:
                for i, record in enumerate(records):
                    if record.snapshot_path:
                        src = self.snapshot_storage / record.snapshot_path
                        if src.exists():
                            zf.write(src, f"snapshots/{record.record_id}.jpg")
            
            # Add clips
            if include_clips:
                for record in records:
                    clip_path = self.export_clip(record)
                    if clip_path:
                        zf.write(clip_path, f"clips/{record.record_id}.mp4")
        
        logger.info(f"Exported {len(records)} records to {archive_path}")
        return str(archive_path)


# Helper function to create record ID
def generate_record_id(
    camera_id: str,
    timestamp: datetime,
    embedding: np.ndarray
) -> str:
    """Generate unique record ID"""
    data = f"{camera_id}:{timestamp.isoformat()}:{embedding[:8].tobytes().hex()}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]
