import asyncio
import io
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.pipeline import FacePipeline
from ..database.services import CriminalService
from ..database.session import Database
from ..database.embedding_index import EmbeddingIndex
from .schemas import (
    CriminalCreate,
    CriminalResponse, 
    CriminalUpdate,
    SearchResponse,
    SearchResult,
    SuccessResponse
)


router = APIRouter(prefix="/api/v1/criminals", tags=["criminals"])


async def get_db_session():
    # This would be injected by dependency injection in main.py
    pass


async def get_criminal_service(db: AsyncSession = Depends(get_db_session)):
    # Initialize embedding index
    embedding_index = EmbeddingIndex(dim=512, metric="cosine", persist_path="./data/faiss.index")
    try:
        embedding_index.load()
    except Exception:
        pass
    return CriminalService(db, embedding_index)


async def get_face_pipeline():
    # This would be injected as a singleton
    pass


@router.post("/", response_model=CriminalResponse)
async def create_criminal(
    criminal_data: CriminalCreate,
    service: CriminalService = Depends(get_criminal_service)
):
    """Create a new criminal record."""
    try:
        criminal = await service.create_criminal(
            name=criminal_data.name,
            age=criminal_data.age,
            gender=criminal_data.gender,
            crime_type=criminal_data.crime_type,
            threat_level=criminal_data.threat_level,
            external_id=criminal_data.external_id,
            notes=criminal_data.notes,
            photo_path=criminal_data.photo_path
        )
        
        response = CriminalResponse.from_orm(criminal)
        response.embedding_count = len(criminal.embeddings) if criminal.embeddings else 0
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create criminal: {str(e)}")


@router.post("/{criminal_id}/upload-photo", response_model=SuccessResponse)
async def upload_criminal_photo(
    criminal_id: int,
    file: UploadFile = File(...),
    service: CriminalService = Depends(get_criminal_service),
    pipeline: FacePipeline = Depends(get_face_pipeline)
):
    """Upload photo for criminal and extract face embeddings."""
    
    # Verify criminal exists
    criminal = await service.get_criminal(criminal_id)
    if not criminal:
        raise HTTPException(status_code=404, detail="Criminal not found")
    
    # Validate file
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process image to extract faces
        result = pipeline.process_frame(image)
        
        if not result["faces"]:
            raise HTTPException(status_code=400, detail="No faces detected in image")
        
        # Save image
        photo_dir = Path("./data/criminal_photos")
        photo_dir.mkdir(parents=True, exist_ok=True)
        photo_path = photo_dir / f"criminal_{criminal_id}_{int(time.time())}.jpg"
        
        cv2.imwrite(str(photo_path), image)
        
        # Extract embeddings
        embeddings = [np.array(face["embedding"]) for face in result["faces"]]
        
        # Add embeddings to database and FAISS index
        await service.add_embeddings(criminal_id, embeddings)
        
        # Update criminal photo path
        criminal.photo_path = str(photo_path)
        await service.db.commit()
        
        return SuccessResponse(
            message=f"Successfully uploaded photo and extracted {len(embeddings)} face embedding(s)"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")


@router.get("/", response_model=List[CriminalResponse])
async def list_criminals(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    threat_level: Optional[str] = Query(None),
    crime_type: Optional[str] = Query(None),
    service: CriminalService = Depends(get_criminal_service)
):
    """List criminals with optional filtering."""
    criminals = await service.list_criminals(
        skip=skip,
        limit=limit,
        threat_level=threat_level,
        crime_type=crime_type
    )
    
    response = []
    for criminal in criminals:
        criminal_response = CriminalResponse.from_orm(criminal)
        criminal_response.embedding_count = len(criminal.embeddings) if criminal.embeddings else 0
        response.append(criminal_response)
    
    return response


@router.get("/{criminal_id}", response_model=CriminalResponse)
async def get_criminal(
    criminal_id: int,
    service: CriminalService = Depends(get_criminal_service)
):
    """Get criminal by ID."""
    criminal = await service.get_criminal(criminal_id)
    if not criminal:
        raise HTTPException(status_code=404, detail="Criminal not found")
    
    response = CriminalResponse.from_orm(criminal)
    response.embedding_count = len(criminal.embeddings) if criminal.embeddings else 0
    return response


@router.delete("/{criminal_id}", response_model=SuccessResponse)
async def delete_criminal(
    criminal_id: int,
    service: CriminalService = Depends(get_criminal_service)
):
    """Delete criminal record."""
    success = await service.delete_criminal(criminal_id)
    if not success:
        raise HTTPException(status_code=404, detail="Criminal not found")
    
    return SuccessResponse(message="Criminal record deleted successfully")


@router.post("/search-by-image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    threshold: float = Query(0.4, ge=0.0, le=1.0),
    max_results: int = Query(10, ge=1, le=100),
    service: CriminalService = Depends(get_criminal_service),
    pipeline: FacePipeline = Depends(get_face_pipeline)
):
    """Search for criminals by uploading an image."""
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    start_time = time.time()
    
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Extract faces and embeddings
        result = pipeline.process_frame(image)
        
        if not result["faces"]:
            return SearchResponse(results=[], query_time_ms=(time.time() - start_time) * 1000)
        
        # Search for matches
        all_results = []
        for face in result["faces"]:
            embedding = np.array([face["embedding"]])
            matches = service.index.search(embedding, k=max_results)
            
            # Process matches
            for embedding_id, similarity in matches[0]:
                if similarity >= threshold and embedding_id != -1:
                    # Get criminal info (this is a simplified version)
                    # In reality, you'd join with the embedding table to get criminal_id
                    all_results.append(SearchResult(
                        criminal_id=embedding_id,  # This would be resolved properly
                        criminal_name="Unknown",  # Would be fetched from DB
                        similarity=similarity,
                        threat_level=None,
                        crime_type=None
                    ))
        
        # Remove duplicates and sort by similarity
        unique_results = {}
        for result in all_results:
            if result.criminal_id not in unique_results or result.similarity > unique_results[result.criminal_id].similarity:
                unique_results[result.criminal_id] = result
        
        final_results = sorted(unique_results.values(), key=lambda x: x.similarity, reverse=True)[:max_results]
        
        query_time = (time.time() - start_time) * 1000
        return SearchResponse(results=final_results, query_time_ms=query_time)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
