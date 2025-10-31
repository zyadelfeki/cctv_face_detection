from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.services import CameraService
from ..database.session import Database
from .schemas import CameraCreate, CameraResponse, SuccessResponse


router = APIRouter(prefix="/api/v1/cameras", tags=["cameras"])


async def get_db_session():
    # This would be injected by dependency injection in main.py
    pass


async def get_camera_service(db: AsyncSession = Depends(get_camera_service)):
    return CameraService(db)


@router.post("/", response_model=CameraResponse)
async def create_camera(
    camera_data: CameraCreate,
    service: CameraService = Depends(get_camera_service)
):
    """Create a new camera."""
    try:
        camera = await service.create_camera(
            name=camera_data.name,
            url=camera_data.url,
            location=camera_data.location,
            active=camera_data.active
        )
        return CameraResponse.from_orm(camera)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create camera: {str(e)}")


@router.get("/", response_model=List[CameraResponse])
async def list_cameras(
    active_only: bool = True,
    service: CameraService = Depends(get_camera_service)
):
    """List cameras."""
    cameras = await service.list_cameras(active_only=active_only)
    return [CameraResponse.from_orm(camera) for camera in cameras]


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(
    camera_id: int,
    service: CameraService = Depends(get_camera_service)
):
    """Get camera by ID."""
    camera = await service.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return CameraResponse.from_orm(camera)
