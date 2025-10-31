from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.services import IncidentService
from .schemas import IncidentResponse, IncidentMatchResponse
from .auth import require_role

router = APIRouter(prefix="/api/v1/incidents", tags=["incidents"])

async def get_db_session():
    pass

async def get_incident_service(db: AsyncSession = Depends(get_db_session)):
    return IncidentService(db)


@router.get("/", response_model=List[IncidentResponse], dependencies=[Depends(require_role("admin", "operator"))])
async def list_incidents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    camera_id: Optional[int] = Query(None),
    with_matches_only: bool = Query(False),
    service: IncidentService = Depends(get_incident_service)
):
    incidents = await service.list_incidents(skip=skip, limit=limit, camera_id=camera_id, with_matches_only=with_matches_only)
    out: List[IncidentResponse] = []
    for inc in incidents:
        out.append(IncidentResponse(
            id=inc.id,
            camera_id=inc.camera_id,
            camera_name=inc.camera.name if inc.camera else "Unknown",
            camera_location=inc.camera.location if inc.camera else None,
            timestamp=inc.timestamp,
            image_path=inc.image_path,
            face_count=inc.face_count,
            matches=[
                IncidentMatchResponse(
                    id=m.id,
                    criminal_id=m.criminal_id,
                    criminal_name=m.criminal.name if m.criminal else "Unknown",
                    similarity=m.similarity,
                    bbox_x1=m.bbox_x1,
                    bbox_y1=m.bbox_y1,
                    bbox_x2=m.bbox_x2,
                    bbox_y2=m.bbox_y2,
                ) for m in inc.matches
            ]
        ))
    return out
