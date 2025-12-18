from datetime import datetime
from typing import List, Optional
import re

from pydantic import BaseModel, Field, field_validator, ConfigDict
from src.utils.validators import (
    StringValidator,
    URLValidator,
    ValidationError as ValidatorError
)


# Criminal schemas
class CriminalBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, description="Criminal name")
    age: Optional[int] = Field(None, ge=0, le=150, description="Age in years")
    gender: Optional[str] = Field(None, max_length=16, description="Gender")
    crime_type: Optional[str] = Field(None, max_length=128, description="Type of crime")
    threat_level: Optional[str] = Field(None, max_length=32, description="Threat level")
    external_id: Optional[str] = Field(None, max_length=64, description="External system ID")
    notes: Optional[str] = Field(None, max_length=2000, description="Additional notes")
    photo_path: Optional[str] = Field(None, max_length=512, description="Path to photo")
    
    @field_validator('name', 'crime_type', 'notes', 'external_id')
    @classmethod
    def sanitize_strings(cls, v: Optional[str]) -> Optional[str]:
        """Sanitize string fields to prevent XSS."""
        if v is None:
            return v
        try:
            return StringValidator.sanitize_string(v, max_length=2000)
        except ValidatorError as e:
            raise ValueError(str(e))
    
    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v: Optional[str]) -> Optional[str]:
        """Validate gender field."""
        if v is None:
            return v
        v = v.strip().lower()
        if v and v not in ('male', 'female', 'other', 'unknown'):
            raise ValueError(
                "Gender must be one of: male, female, other, unknown"
            )
        return v
    
    @field_validator('threat_level')
    @classmethod
    def validate_threat_level(cls, v: Optional[str]) -> Optional[str]:
        """Validate threat level."""
        if v is None:
            return v
        v = v.strip().lower()
        if v and v not in ('low', 'medium', 'high', 'critical'):
            raise ValueError(
                "Threat level must be one of: low, medium, high, critical"
            )
        return v


class CriminalCreate(CriminalBase):
    pass


class CriminalUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=128)
    age: Optional[int] = Field(None, ge=0, le=150)
    gender: Optional[str] = Field(None, max_length=16)
    crime_type: Optional[str] = Field(None, max_length=128)
    threat_level: Optional[str] = Field(None, max_length=32)
    external_id: Optional[str] = Field(None, max_length=64)
    notes: Optional[str] = Field(None, max_length=2000)
    photo_path: Optional[str] = Field(None, max_length=512)
    
    @field_validator('name', 'crime_type', 'notes', 'external_id')
    @classmethod
    def sanitize_strings(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        try:
            return StringValidator.sanitize_string(v, max_length=2000)
        except ValidatorError as e:
            raise ValueError(str(e))
    
    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip().lower()
        if v and v not in ('male', 'female', 'other', 'unknown'):
            raise ValueError(
                "Gender must be one of: male, female, other, unknown"
            )
        return v
    
    @field_validator('threat_level')
    @classmethod
    def validate_threat_level(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip().lower()
        if v and v not in ('low', 'medium', 'high', 'critical'):
            raise ValueError(
                "Threat level must be one of: low, medium, high, critical"
            )
        return v


class CriminalResponse(CriminalBase):
    id: int
    created_at: datetime
    updated_at: datetime
    embedding_count: int = 0
    
    model_config = ConfigDict(from_attributes=True)


# Camera schemas
class CameraBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, description="Camera name")
    url: str = Field(..., min_length=1, max_length=512, description="RTSP URL")
    location: Optional[str] = Field(None, max_length=256, description="Camera location")
    active: bool = Field(True, description="Whether camera is active")
    
    @field_validator('name', 'location')
    @classmethod
    def sanitize_strings(cls, v: Optional[str]) -> Optional[str]:
        """Sanitize string fields."""
        if v is None:
            return v
        try:
            return StringValidator.sanitize_string(v, max_length=256)
        except ValidatorError as e:
            raise ValueError(str(e))
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate RTSP URL and prevent SSRF."""
        try:
            # Allow private IPs for cameras (internal network)
            parsed = URLValidator.validate_rtsp_url(v, allow_private=True)
            return v
        except ValidatorError as e:
            raise ValueError(f"Invalid RTSP URL: {e}")


class CameraCreate(CameraBase):
    pass


class CameraResponse(CameraBase):
    id: int
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# Incident schemas
class IncidentMatchResponse(BaseModel):
    id: int
    criminal_id: int
    criminal_name: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    bbox_x1: int = Field(..., ge=0)
    bbox_y1: int = Field(..., ge=0)
    bbox_x2: int = Field(..., ge=0)
    bbox_y2: int = Field(..., ge=0)
    
    model_config = ConfigDict(from_attributes=True)


class IncidentResponse(BaseModel):
    id: int
    camera_id: int
    camera_name: str
    camera_location: Optional[str]
    timestamp: datetime
    image_path: Optional[str]
    face_count: int = Field(..., ge=0)
    matches: List[IncidentMatchResponse] = []
    
    model_config = ConfigDict(from_attributes=True)


# Search schemas
class SearchResult(BaseModel):
    criminal_id: int
    criminal_name: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    threat_level: Optional[str]
    crime_type: Optional[str]


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query_time_ms: float = Field(..., ge=0.0)


class SearchParams(BaseModel):
    """Parameters for face search."""
    threshold: float = Field(0.4, ge=0.0, le=1.0, description="Similarity threshold")
    max_results: int = Field(10, ge=1, le=100, description="Maximum results")
    threat_level: Optional[str] = Field(None, description="Filter by threat level")
    
    @field_validator('threat_level')
    @classmethod
    def validate_threat_level(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip().lower()
        if v not in ('low', 'medium', 'high', 'critical'):
            raise ValueError(
                "Threat level must be one of: low, medium, high, critical"
            )
        return v


# Generic response schemas
class SuccessResponse(BaseModel):
    success: bool = True
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None


# Pagination
class PaginationParams(BaseModel):
    skip: int = Field(0, ge=0, le=10000, description="Number of items to skip")
    limit: int = Field(100, ge=1, le=1000, description="Number of items to return")


class PaginatedResponse(BaseModel):
    items: List
    total: int = Field(..., ge=0)
    skip: int = Field(..., ge=0)
    limit: int = Field(..., ge=1)
    has_next: bool


# File upload validation
class FileUploadParams(BaseModel):
    """Parameters for file uploads."""
    max_size_mb: float = Field(10.0, ge=0.1, le=100.0)
    allowed_types: List[str] = Field(
        default=['image/jpeg', 'image/png', 'image/jpg'],
        description="Allowed MIME types"
    )


# Configuration schemas
class SystemConfig(BaseModel):
    """System configuration parameters."""
    max_upload_size_mb: float = Field(10.0, ge=0.1, le=100.0)
    max_concurrent_streams: int = Field(10, ge=1, le=100)
    face_detection_confidence: float = Field(0.7, ge=0.0, le=1.0)
    face_recognition_threshold: float = Field(0.4, ge=0.0, le=1.0)
    
    @field_validator('face_detection_confidence', 'face_recognition_threshold')
    @classmethod
    def validate_probability(cls, v: float) -> float:
        """Ensure value is valid probability."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Probability must be between 0.0 and 1.0")
        return v


# Health check schemas
class HealthStatus(BaseModel):
    status: str = Field(..., pattern=r'^(ok|degraded|error)$')
    timestamp: datetime
    components: dict = {}
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v: str) -> str:
        v = v.lower()
        if v not in ('ok', 'degraded', 'error'):
            raise ValueError("Status must be: ok, degraded, or error")
        return v


# Metrics schemas
class SystemMetrics(BaseModel):
    """System performance metrics."""
    cpu_percent: float = Field(..., ge=0.0, le=100.0)
    memory_percent: float = Field(..., ge=0.0, le=100.0)
    disk_percent: float = Field(..., ge=0.0, le=100.0)
    active_cameras: int = Field(..., ge=0)
    total_detections_today: int = Field(..., ge=0)
    avg_processing_time_ms: float = Field(..., ge=0.0)
