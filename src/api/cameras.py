from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.services import CameraService
from .schemas import CameraCreate, CameraResponse
from .auth import require_role

router = APIRouter(prefix="/api/v1/cameras", tags=["cameras"])

async def get_db_session():
    pass

async def get_camera_service(db: AsyncSession = Depends(get_db_session)):
    return CameraService(db)


@router.post("/", response_model=CameraResponse, dependencies=[Depends(require_role("admin"))])
async def create_camera(camera_data: CameraCreate, service: CameraService = Depends(get_camera_service)):
    camera = await service.create_camera(name=camera_data.name, url=camera_data.url, location=camera_data.location, active=camera_data.active)
    return CameraResponse.from_orm(camera)


@router.get("/", response_model=List[CameraResponse], dependencies=[Depends(require_role("admin", "operator"))])
async def list_cameras(active_only: bool = True, service: CameraService = Depends(get_camera_service)):
    cams = await service.list_cameras(active_only=active_only)
    return [CameraResponse.from_orm(c) for c in cams]
