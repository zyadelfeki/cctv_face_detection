from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.services import IncidentService
from ..database.session import Database
from .schemas import IncidentResponse, IncidentMatchResponse


router = APIRouter(prefix="/api/v1/incidents", tags=["incidents"])


async def get_db_session():
    # This would be injected by dependency injection in main.py
    pass


async def get_incident_service(db: AsyncSession = Depends(get_db_session)):
    return IncidentService(db)


@router.get("/", response_model=List[IncidentResponse])
async def list_incidents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    camera_id: Optional[int] = Query(None),
    with_matches_only: bool = Query(False),
    service: IncidentService = Depends(get_incident_service)
):
    """List incidents with optional filtering."""
    incidents = await service.list_incidents(
        skip=skip,
        limit=limit,
        camera_id=camera_id,
        with_matches_only=with_matches_only
    )
    
    response = []
    for incident in incidents:
        incident_response = IncidentResponse(
            id=incident.id,
            camera_id=incident.camera_id,
            camera_name=incident.camera.name if incident.camera else "Unknown",
            camera_location=incident.camera.location if incident.camera else None,
            timestamp=incident.timestamp,
            image_path=incident.image_path,
            face_count=incident.face_count,
            matches=[
                IncidentMatchResponse(
                    id=match.id,
                    criminal_id=match.criminal_id,
                    criminal_name=match.criminal.name if match.criminal else "Unknown",
                    similarity=match.similarity,
                    bbox_x1=match.bbox_x1,
                    bbox_y1=match.bbox_y1,
                    bbox_x2=match.bbox_x2,
                    bbox_y2=match.bbox_y2
                )
                for match in incident.matches
            ]
        )
        response.append(incident_response)
    
    return response


@router.get("/{incident_id}", response_model=IncidentResponse)
async def get_incident(
    incident_id: int,
    service: IncidentService = Depends(get_incident_service)
):
    """Get incident by ID with matches."""
    incidents = await service.list_incidents(skip=0, limit=1)
    # This is a simplified implementation - would need a proper get_by_id method
    incident = next((i for i in incidents if i.id == incident_id), None)
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    return IncidentResponse(
        id=incident.id,
        camera_id=incident.camera_id,
        camera_name=incident.camera.name if incident.camera else "Unknown",
        camera_location=incident.camera.location if incident.camera else None,
        timestamp=incident.timestamp,
        image_path=incident.image_path,
        face_count=incident.face_count,
        matches=[
            IncidentMatchResponse(
                id=match.id,
                criminal_id=match.criminal_id,
                criminal_name=match.criminal.name if match.criminal else "Unknown",
                similarity=match.similarity,
                bbox_x1=match.bbox_x1,
                bbox_y1=match.bbox_y1,
                bbox_x2=match.bbox_x2,
                bbox_y2=match.bbox_y2
            )
            for match in incident.matches
        ]
    )
