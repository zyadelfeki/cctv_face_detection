from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# Criminal schemas
class CriminalBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    age: Optional[int] = Field(None, ge=0, le=150)
    gender: Optional[str] = Field(None, max_length=16)
    crime_type: Optional[str] = Field(None, max_length=128)
    threat_level: Optional[str] = Field(None, max_length=32)
    external_id: Optional[str] = Field(None, max_length=64)
    notes: Optional[str] = None
    photo_path: Optional[str] = None


class CriminalCreate(CriminalBase):
    pass


class CriminalUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=128)
    age: Optional[int] = Field(None, ge=0, le=150)
    gender: Optional[str] = Field(None, max_length=16)
    crime_type: Optional[str] = Field(None, max_length=128)
    threat_level: Optional[str] = Field(None, max_length=32)
    external_id: Optional[str] = Field(None, max_length=64)
    notes: Optional[str] = None
    photo_path: Optional[str] = None


class CriminalResponse(CriminalBase):
    id: int
    created_at: datetime
    updated_at: datetime
    embedding_count: int = 0

    class Config:
        from_attributes = True


# Camera schemas
class CameraBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    url: str = Field(..., min_length=1)
    location: Optional[str] = Field(None, max_length=256)
    active: bool = True


class CameraCreate(CameraBase):
    pass


class CameraResponse(CameraBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# Incident schemas
class IncidentMatchResponse(BaseModel):
    id: int
    criminal_id: int
    criminal_name: str
    similarity: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int

    class Config:
        from_attributes = True


class IncidentResponse(BaseModel):
    id: int
    camera_id: int
    camera_name: str
    camera_location: Optional[str]
    timestamp: datetime
    image_path: Optional[str]
    face_count: int
    matches: List[IncidentMatchResponse] = []

    class Config:
        from_attributes = True


# Search schemas
class SearchResult(BaseModel):
    criminal_id: int
    criminal_name: str
    similarity: float
    threat_level: Optional[str]
    crime_type: Optional[str]


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query_time_ms: float


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
    skip: int = Field(0, ge=0)
    limit: int = Field(100, ge=1, le=1000)


class PaginatedResponse(BaseModel):
    items: List
    total: int
    skip: int
    limit: int
    has_next: bool
