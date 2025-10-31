from datetime import datetime
from typing import List, Optional

import json
import numpy as np
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import Criminal, Embedding, Camera, Incident, IncidentMatch
from .embedding_index import EmbeddingIndex
from loguru import logger


class CriminalService:
    def __init__(self, db_session: AsyncSession, embedding_index: EmbeddingIndex):
        self.db = db_session
        self.index = embedding_index

    async def create_criminal(
        self,
        name: str,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        crime_type: Optional[str] = None,
        threat_level: Optional[str] = None,
        external_id: Optional[str] = None,
        notes: Optional[str] = None,
        photo_path: Optional[str] = None,
        embeddings: Optional[List[np.ndarray]] = None
    ) -> Criminal:
        """Create criminal record with optional face embeddings."""
        
        criminal = Criminal(
            name=name,
            age=age,
            gender=gender,
            crime_type=crime_type,
            threat_level=threat_level,
            external_id=external_id,
            notes=notes,
            photo_path=photo_path
        )
        
        self.db.add(criminal)
        await self.db.flush()  # Get the ID
        
        if embeddings:
            await self.add_embeddings(criminal.id, embeddings)
            
        await self.db.commit()
        return criminal

    async def add_embeddings(self, criminal_id: int, embeddings: List[np.ndarray]):
        """Add face embeddings for a criminal."""
        embedding_records = []
        vectors_for_faiss = []
        ids_for_faiss = []
        
        for emb in embeddings:
            # Store embedding as JSON string in DB (backup)
            embedding_record = Embedding(
                criminal_id=criminal_id,
                vector=json.dumps(emb.tolist()),
                dim=len(emb)
            )
            self.db.add(embedding_record)
            embedding_records.append(embedding_record)
            
        await self.db.flush()  # Get embedding IDs
        
        # Add to FAISS index
        for i, emb in enumerate(embeddings):
            vectors_for_faiss.append(emb)
            ids_for_faiss.append(embedding_records[i].id)
            
        if vectors_for_faiss:
            vectors_array = np.vstack(vectors_for_faiss)
            self.index.add(vectors_array, ids_for_faiss)
            self.index.save()
            
        await self.db.commit()
        return embedding_records

    async def get_criminal(self, criminal_id: int) -> Optional[Criminal]:
        """Get criminal by ID with embeddings."""
        query = select(Criminal).options(selectinload(Criminal.embeddings)).where(Criminal.id == criminal_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def list_criminals(
        self, 
        skip: int = 0, 
        limit: int = 100,
        threat_level: Optional[str] = None,
        crime_type: Optional[str] = None
    ) -> List[Criminal]:
        """List criminals with optional filters."""
        query = select(Criminal)
        
        if threat_level:
            query = query.where(Criminal.threat_level == threat_level)
        if crime_type:
            query = query.where(Criminal.crime_type == crime_type)
            
        query = query.offset(skip).limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def delete_criminal(self, criminal_id: int) -> bool:
        """Delete criminal and associated embeddings."""
        # Get embedding IDs to remove from FAISS
        emb_query = select(Embedding.id).where(Embedding.criminal_id == criminal_id)
        emb_result = await self.db.execute(emb_query)
        embedding_ids = [row[0] for row in emb_result.all()]
        
        # Delete from database (cascades to embeddings)
        delete_query = delete(Criminal).where(Criminal.id == criminal_id)
        result = await self.db.execute(delete_query)
        await self.db.commit()
        
        # TODO: Remove from FAISS index (requires rebuilding index)
        # For now, mark as deleted or rebuild index periodically
        
        return result.rowcount > 0


class IncidentService:
    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def create_incident(
        self,
        camera_id: int,
        face_count: int = 0,
        image_path: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> Incident:
        """Create new incident record."""
        incident = Incident(
            camera_id=camera_id,
            face_count=face_count,
            image_path=image_path,
            timestamp=timestamp or datetime.utcnow()
        )
        
        self.db.add(incident)
        await self.db.flush()
        return incident

    async def add_match(
        self,
        incident_id: int,
        criminal_id: int,
        similarity: float,
        bbox: tuple[int, int, int, int]
    ) -> IncidentMatch:
        """Add match to incident."""
        x1, y1, x2, y2 = bbox
        match = IncidentMatch(
            incident_id=incident_id,
            criminal_id=criminal_id,
            similarity=similarity,
            bbox_x1=x1,
            bbox_y1=y1,
            bbox_x2=x2,
            bbox_y2=y2
        )
        
        self.db.add(match)
        await self.db.flush()
        return match

    async def list_incidents(
        self,
        skip: int = 0,
        limit: int = 100,
        camera_id: Optional[int] = None,
        with_matches_only: bool = False
    ) -> List[Incident]:
        """List incidents with optional filters."""
        query = select(Incident).options(
            selectinload(Incident.camera),
            selectinload(Incident.matches).selectinload(IncidentMatch.criminal)
        )
        
        if camera_id:
            query = query.where(Incident.camera_id == camera_id)
        if with_matches_only:
            query = query.join(IncidentMatch)
            
        query = query.order_by(Incident.timestamp.desc()).offset(skip).limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().unique().all())


class CameraService:
    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def create_camera(
        self,
        name: str,
        url: str,
        location: Optional[str] = None,
        active: bool = True
    ) -> Camera:
        """Create camera record."""
        camera = Camera(
            name=name,
            url=url,
            location=location,
            active=active
        )
        
        self.db.add(camera)
        await self.db.commit()
        return camera

    async def list_cameras(self, active_only: bool = True) -> List[Camera]:
        """List cameras."""
        query = select(Camera)
        if active_only:
            query = query.where(Camera.active == True)
        
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_camera(self, camera_id: int) -> Optional[Camera]:
        """Get camera by ID."""
        query = select(Camera).where(Camera.id == camera_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
