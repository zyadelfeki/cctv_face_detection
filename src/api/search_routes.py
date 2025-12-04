"""
Face Search API Routes

REST endpoints for face database search:
- Similarity search by photo
- Time-range queries
- Multi-filter search
- Clip/snapshot export
- Person timeline

Author: CCTV Face Detection System
"""

import asyncio
import io
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import numpy as np
import cv2

from ..core.face_database import (
    ClipExporter,
    FaceDatabase,
    FaceRecord,
    SearchQuery,
    SearchResult,
    generate_record_id
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/faces", tags=["face-search"])

# Global instances
face_db: Optional[FaceDatabase] = None
clip_exporter: Optional[ClipExporter] = None

# Placeholder for face embedding model
embedding_model = None


# ============ Pydantic Models ============

class FaceSearchRequest(BaseModel):
    """Request for face search"""
    # Similarity search (base64 or reference)
    reference_record_id: Optional[str] = None
    
    # Person filter
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    
    # Time filter
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Camera filter
    camera_ids: Optional[List[str]] = None
    
    # Attribute filters
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    is_known: Optional[bool] = None
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    gender: Optional[str] = Field(default=None, pattern="^(male|female)$")
    emotion: Optional[str] = None
    
    # Similarity threshold
    similarity_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    
    # Pagination
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    
    # Sort
    sort_by: str = Field(default="timestamp", pattern="^(timestamp|confidence|similarity)$")
    sort_desc: bool = True


class FaceRecordResponse(BaseModel):
    """Response for a single face record"""
    record_id: str
    person_id: Optional[str]
    person_name: Optional[str]
    confidence: float
    camera_id: str
    camera_name: str
    timestamp: datetime
    bounding_box: List[int]
    snapshot_url: Optional[str]
    video_clip_url: Optional[str]
    attributes: Dict
    is_known: bool
    liveness_score: Optional[float]
    similarity_score: Optional[float] = None
    rank: int = 0


class SearchResponse(BaseModel):
    """Response for face search"""
    query_time_ms: float
    total_results: int
    offset: int
    limit: int
    results: List[FaceRecordResponse]


class TimelineResponse(BaseModel):
    """Response for person timeline"""
    person_id: str
    person_name: Optional[str]
    total_appearances: int
    cameras_visited: List[str]
    first_seen: datetime
    last_seen: datetime
    appearances: List[FaceRecordResponse]


class CameraActivityResponse(BaseModel):
    """Response for camera activity"""
    camera_id: str
    start_time: datetime
    end_time: datetime
    total_detections: int
    known_faces: int
    unknown_faces: int
    unique_persons: int
    hourly_breakdown: Dict[str, int]


class ExportRequest(BaseModel):
    """Request to export search results"""
    record_ids: List[str]
    include_snapshots: bool = True
    include_clips: bool = False
    format: str = Field(default="zip", pattern="^(zip|json)$")


class DatabaseStatsResponse(BaseModel):
    """Database statistics"""
    total_records: int
    unique_persons: int
    cameras: int
    earliest_record: Optional[str]
    latest_record: Optional[str]
    index_type: str


# ============ Lifecycle ============

async def init_face_search(
    storage_path: str = "data/face_db",
    video_storage: str = "data/videos",
    snapshot_storage: str = "data/snapshots",
    export_path: str = "exports"
):
    """Initialize face search services on startup"""
    global face_db, clip_exporter
    
    face_db = FaceDatabase(storage_path=storage_path)
    clip_exporter = ClipExporter(
        video_storage_path=video_storage,
        snapshot_storage_path=snapshot_storage,
        export_path=export_path
    )
    
    logger.info("Face search services initialized")


# ============ Helper Functions ============

def record_to_response(
    record: FaceRecord,
    similarity: Optional[float] = None,
    rank: int = 0,
    base_url: str = "/api/faces"
) -> FaceRecordResponse:
    """Convert FaceRecord to API response"""
    return FaceRecordResponse(
        record_id=record.record_id,
        person_id=record.person_id,
        person_name=record.person_name,
        confidence=record.confidence,
        camera_id=record.camera_id,
        camera_name=record.camera_name,
        timestamp=record.timestamp,
        bounding_box=list(record.bounding_box),
        snapshot_url=f"{base_url}/snapshot/{record.record_id}" if record.snapshot_path else None,
        video_clip_url=f"{base_url}/clip/{record.record_id}" if record.video_clip_path else None,
        attributes=record.attributes,
        is_known=record.is_known,
        liveness_score=record.liveness_score,
        similarity_score=similarity,
        rank=rank
    )


async def get_embedding_from_image(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract face embedding from image.
    
    This would use your face recognition model.
    Placeholder implementation.
    """
    # TODO: Integrate with actual face embedding model
    # For now, return mock embedding
    return np.random.rand(512).astype(np.float32)


# ============ Search Endpoints ============

@router.post("/search", response_model=SearchResponse)
async def search_faces(request: FaceSearchRequest):
    """
    Search the face database with multiple filters.
    
    Supports:
    - Reference-based similarity search
    - Person ID/name filtering
    - Time range queries
    - Camera filtering
    - Demographic filtering (age, gender, emotion)
    """
    if not face_db:
        raise HTTPException(status_code=503, detail="Face database not initialized")
    
    import time
    start = time.time()
    
    # Build query
    query_embedding = None
    if request.reference_record_id:
        ref_record = face_db.records.get(request.reference_record_id)
        if ref_record:
            query_embedding = ref_record.embedding
    
    query = SearchQuery(
        query_embedding=query_embedding,
        similarity_threshold=request.similarity_threshold,
        person_id=request.person_id,
        person_name=request.person_name,
        start_time=request.start_time,
        end_time=request.end_time,
        camera_ids=request.camera_ids,
        min_confidence=request.min_confidence,
        is_known=request.is_known,
        min_age=request.min_age,
        max_age=request.max_age,
        gender=request.gender,
        emotion=request.emotion,
        limit=request.limit,
        offset=request.offset,
        sort_by=request.sort_by,
        sort_desc=request.sort_desc
    )
    
    results = face_db.search(query)
    
    query_time = (time.time() - start) * 1000
    
    return SearchResponse(
        query_time_ms=query_time,
        total_results=len(results),
        offset=request.offset,
        limit=request.limit,
        results=[
            record_to_response(r.record, r.similarity_score, r.rank)
            for r in results
        ]
    )


@router.post("/search/photo")
async def search_by_photo(
    photo: UploadFile = File(..., description="Photo to search for"),
    similarity_threshold: float = Query(default=0.6, ge=0.0, le=1.0),
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    camera_ids: Optional[str] = Query(default=None, description="Comma-separated camera IDs"),
    limit: int = Query(default=50, ge=1, le=500)
):
    """
    Search by uploading a photo.
    
    Upload an image and find all matching face detections.
    """
    if not face_db:
        raise HTTPException(status_code=503, detail="Face database not initialized")
    
    # Read and decode image
    contents = await photo.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Extract face embedding
    embedding = await get_embedding_from_image(img)
    if embedding is None:
        raise HTTPException(status_code=400, detail="No face detected in image")
    
    # Parse camera IDs
    cam_ids = camera_ids.split(",") if camera_ids else None
    
    # Search
    import time
    start = time.time()
    
    results = face_db.search_by_photo(
        photo_embedding=embedding,
        threshold=similarity_threshold,
        limit=limit,
        time_range=(start_time, end_time) if start_time or end_time else None,
        camera_ids=cam_ids
    )
    
    query_time = (time.time() - start) * 1000
    
    return {
        "query_time_ms": query_time,
        "total_results": len(results),
        "results": [
            record_to_response(r.record, r.similarity_score, r.rank)
            for r in results
        ]
    }


@router.get("/timeline/{person_id}", response_model=TimelineResponse)
async def get_person_timeline(
    person_id: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
):
    """
    Get timeline of a person's appearances.
    
    Shows movement through different cameras over time.
    """
    if not face_db:
        raise HTTPException(status_code=503, detail="Face database not initialized")
    
    records = face_db.get_person_timeline(
        person_id=person_id,
        start_time=start_time,
        end_time=end_time
    )
    
    if not records:
        raise HTTPException(status_code=404, detail="Person not found")
    
    cameras = list(set(r.camera_id for r in records))
    
    return TimelineResponse(
        person_id=person_id,
        person_name=records[0].person_name if records else None,
        total_appearances=len(records),
        cameras_visited=cameras,
        first_seen=records[0].timestamp,
        last_seen=records[-1].timestamp,
        appearances=[record_to_response(r, rank=i+1) for i, r in enumerate(records)]
    )


@router.get("/camera/{camera_id}/activity")
async def get_camera_activity(
    camera_id: str,
    start_time: datetime = Query(..., description="Start of time range"),
    end_time: datetime = Query(..., description="End of time range")
):
    """
    Get activity summary for a camera.
    
    Returns detection counts and breakdown by hour.
    """
    if not face_db:
        raise HTTPException(status_code=503, detail="Face database not initialized")
    
    activity = face_db.get_camera_activity(camera_id, start_time, end_time)
    
    # Calculate hourly breakdown
    hourly: Dict[str, int] = {}
    for record_dict in activity.get("records", []):
        ts = datetime.fromisoformat(record_dict["timestamp"])
        hour_key = ts.strftime("%Y-%m-%d %H:00")
        hourly[hour_key] = hourly.get(hour_key, 0) + 1
    
    return CameraActivityResponse(
        camera_id=camera_id,
        start_time=start_time,
        end_time=end_time,
        total_detections=activity["total_detections"],
        known_faces=activity["known_faces"],
        unknown_faces=activity["unknown_faces"],
        unique_persons=activity["unique_persons"],
        hourly_breakdown=hourly
    )


# ============ Record Endpoints ============

@router.get("/record/{record_id}")
async def get_record(record_id: str):
    """Get a single face record by ID"""
    if not face_db:
        raise HTTPException(status_code=503, detail="Face database not initialized")
    
    record = face_db.records.get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    return record_to_response(record)


@router.get("/snapshot/{record_id}")
async def get_snapshot(
    record_id: str,
    draw_bbox: bool = Query(default=False, description="Draw bounding box on image")
):
    """Get snapshot image for a record"""
    if not face_db or not clip_exporter:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    record = face_db.records.get(record_id)
    if not record or not record.snapshot_path:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    
    path = clip_exporter.export_snapshot(record, draw_bbox=draw_bbox)
    if not path:
        raise HTTPException(status_code=404, detail="Snapshot file not found")
    
    return FileResponse(path, media_type="image/jpeg")


@router.get("/clip/{record_id}")
async def get_video_clip(
    record_id: str,
    before_seconds: float = Query(default=5.0, ge=0, le=30),
    after_seconds: float = Query(default=5.0, ge=0, le=30)
):
    """
    Get video clip around detection time.
    
    Extracts clip from stored video footage.
    """
    if not face_db or not clip_exporter:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    record = face_db.records.get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    path = clip_exporter.export_clip(
        record,
        before_seconds=before_seconds,
        after_seconds=after_seconds
    )
    
    if not path:
        raise HTTPException(status_code=404, detail="Video clip not available")
    
    return FileResponse(path, media_type="video/mp4")


# ============ Export Endpoints ============

@router.post("/export")
async def export_records(request: ExportRequest):
    """
    Export multiple records as ZIP archive.
    
    Includes metadata, snapshots, and optionally video clips.
    """
    if not face_db or not clip_exporter:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    # Get records
    records = []
    for rid in request.record_ids:
        record = face_db.records.get(rid)
        if record:
            records.append(record)
    
    if not records:
        raise HTTPException(status_code=404, detail="No valid records found")
    
    if request.format == "json":
        # Return JSON directly
        return {
            "export_time": datetime.now().isoformat(),
            "record_count": len(records),
            "records": [r.to_dict() for r in records]
        }
    else:
        # Create ZIP archive
        archive_path = clip_exporter.export_batch(
            records,
            include_snapshots=request.include_snapshots,
            include_clips=request.include_clips
        )
        
        return FileResponse(
            archive_path,
            media_type="application/zip",
            filename=f"face_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        )


# ============ Statistics ============

@router.get("/stats", response_model=DatabaseStatsResponse)
async def get_database_stats():
    """Get database statistics"""
    if not face_db:
        raise HTTPException(status_code=503, detail="Face database not initialized")
    
    stats = face_db.get_stats()
    
    return DatabaseStatsResponse(
        total_records=stats["total_records"],
        unique_persons=stats["unique_persons"],
        cameras=stats["cameras"],
        earliest_record=stats["date_range"]["earliest"],
        latest_record=stats["date_range"]["latest"],
        index_type="FAISS" if hasattr(face_db.index, 'index') else "Numpy"
    )


@router.get("/persons")
async def list_persons(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0)
):
    """List all known persons in database"""
    if not face_db:
        raise HTTPException(status_code=503, detail="Face database not initialized")
    
    persons = []
    for person_id, record_ids in list(face_db.by_person.items())[offset:offset+limit]:
        # Get first record for name
        first_record = face_db.records.get(record_ids[0]) if record_ids else None
        
        persons.append({
            "person_id": person_id,
            "person_name": first_record.person_name if first_record else None,
            "detection_count": len(record_ids),
            "first_seen": min(face_db.records[rid].timestamp for rid in record_ids if rid in face_db.records).isoformat(),
            "last_seen": max(face_db.records[rid].timestamp for rid in record_ids if rid in face_db.records).isoformat()
        })
    
    return {
        "total": len(face_db.by_person),
        "offset": offset,
        "limit": limit,
        "persons": persons
    }


@router.get("/cameras")
async def list_cameras():
    """List all cameras with detection counts"""
    if not face_db:
        raise HTTPException(status_code=503, detail="Face database not initialized")
    
    cameras = []
    for camera_id, record_ids in face_db.by_camera.items():
        first_record = face_db.records.get(record_ids[0]) if record_ids else None
        
        cameras.append({
            "camera_id": camera_id,
            "camera_name": first_record.camera_name if first_record else camera_id,
            "detection_count": len(record_ids)
        })
    
    return cameras


# ============ Quick Queries ============

@router.get("/recent")
async def get_recent_detections(
    limit: int = Query(default=20, ge=1, le=100),
    camera_id: Optional[str] = None,
    is_known: Optional[bool] = None
):
    """Get most recent face detections"""
    if not face_db:
        raise HTTPException(status_code=503, detail="Face database not initialized")
    
    query = SearchQuery(
        camera_ids=[camera_id] if camera_id else None,
        is_known=is_known,
        limit=limit,
        sort_by="timestamp",
        sort_desc=True
    )
    
    results = face_db.search(query)
    
    return [record_to_response(r.record, rank=r.rank) for r in results]


@router.get("/unknown")
async def get_unknown_faces(
    limit: int = Query(default=50, ge=1, le=200),
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
):
    """Get all unknown/unidentified faces"""
    if not face_db:
        raise HTTPException(status_code=503, detail="Face database not initialized")
    
    query = SearchQuery(
        is_known=False,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        sort_by="timestamp",
        sort_desc=True
    )
    
    results = face_db.search(query)
    
    return {
        "total": len(results),
        "unknown_faces": [record_to_response(r.record, rank=r.rank) for r in results]
    }
