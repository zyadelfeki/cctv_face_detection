from fastapi import APIRouter, Depends, HTTPException, Query, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import cv2
import numpy as np
from pathlib import Path
import time

from ..database.services import CriminalService
from ..database.session import Database
from ..database.embedding_index import EmbeddingIndex
from ..core.pipeline import FacePipeline
from .schemas import (
    CriminalCreate,
    CriminalResponse,
    CriminalUpdate,
    SearchResponse,
    SearchResult,
    SuccessResponse,
)
from .auth import AuthService, require_role, get_current_user


router = APIRouter(prefix="/api/v1/criminals", tags=["criminals"])

# Dependencies are provided by main app via monkey-patched functions
async def get_db_session():
    pass

async def get_face_pipeline():
    pass

async def get_criminal_service(db: AsyncSession = Depends(get_db_session)):
    embedding_index = EmbeddingIndex(dim=512, metric="cosine", persist_path="./data/faiss.index")
    try:
        embedding_index.load()
    except Exception:
        pass
    return CriminalService(db, embedding_index)


@router.post("/", response_model=CriminalResponse, dependencies=[Depends(require_role("admin"))])
async def create_criminal(
    criminal_data: CriminalCreate,
    service: CriminalService = Depends(get_criminal_service)
):
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
        resp = CriminalResponse.from_orm(criminal)
        resp.embedding_count = len(criminal.embeddings) if criminal.embeddings else 0
        return resp
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create criminal: {str(e)}")


@router.post("/{criminal_id}/upload-photo", response_model=SuccessResponse, dependencies=[Depends(require_role("admin"))])
async def upload_criminal_photo(
    criminal_id: int,
    file: UploadFile = File(...),
    service: CriminalService = Depends(get_criminal_service),
    pipeline: FacePipeline = Depends(get_face_pipeline)
):
    criminal = await service.get_criminal(criminal_id)
    if not criminal:
        raise HTTPException(status_code=404, detail="Criminal not found")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = pipeline.process_frame(image)
    if not result["faces"]:
        raise HTTPException(status_code=400, detail="No faces detected in image")

    photo_dir = Path("./data/criminal_photos")
    photo_dir.mkdir(parents=True, exist_ok=True)
    photo_path = photo_dir / f"criminal_{criminal_id}_{int(time.time())}.jpg"
    cv2.imwrite(str(photo_path), image)

    embeddings = [np.array(face["embedding"]) for face in result["faces"]]
    await service.add_embeddings(criminal_id, embeddings)

    criminal.photo_path = str(photo_path)
    await service.db.commit()

    return SuccessResponse(message=f"Successfully uploaded photo and extracted {len(embeddings)} embedding(s)")


@router.get("/", response_model=List[CriminalResponse], dependencies=[Depends(require_role("admin", "operator"))])
async def list_criminals(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    threat_level: Optional[str] = Query(None),
    crime_type: Optional[str] = Query(None),
    service: CriminalService = Depends(get_criminal_service)
):
    criminals = await service.list_criminals(skip=skip, limit=limit, threat_level=threat_level, crime_type=crime_type)
    out: List[CriminalResponse] = []
    for c in criminals:
        cr = CriminalResponse.from_orm(c)
        cr.embedding_count = len(c.embeddings) if c.embeddings else 0
        out.append(cr)
    return out


@router.get("/{criminal_id}", response_model=CriminalResponse, dependencies=[Depends(require_role("admin", "operator"))])
async def get_criminal(criminal_id: int, service: CriminalService = Depends(get_criminal_service)):
    c = await service.get_criminal(criminal_id)
    if not c:
        raise HTTPException(status_code=404, detail="Criminal not found")
    resp = CriminalResponse.from_orm(c)
    resp.embedding_count = len(c.embeddings) if c.embeddings else 0
    return resp


@router.delete("/{criminal_id}", response_model=SuccessResponse, dependencies=[Depends(require_role("admin"))])
async def delete_criminal(criminal_id: int, service: CriminalService = Depends(get_criminal_service)):
    success = await service.delete_criminal(criminal_id)
    if not success:
        raise HTTPException(status_code=404, detail="Criminal not found")
    return SuccessResponse(message="Criminal deleted")


@router.post("/search-by-image", response_model=SearchResponse, dependencies=[Depends(require_role("admin", "operator"))])
async def search_by_image(
    file: UploadFile = File(...),
    threshold: float = Query(0.4, ge=0.0, le=1.0),
    max_results: int = Query(10, ge=1, le=100),
    service: CriminalService = Depends(get_criminal_service),
    pipeline: FacePipeline = Depends(get_face_pipeline)
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = pipeline.process_frame(image)
    if not result["faces"]:
        return SearchResponse(results=[], query_time_ms=0.0)

    all_results: List[SearchResult] = []
    for face in result["faces"]:
        embedding = np.array([face["embedding"]], dtype=np.float32)
        matches = service.index.search(embedding, k=max_results)
        for embedding_id, similarity in matches[0]:
            if similarity >= threshold and embedding_id != -1:
                # NOTE: In a follow-up, resolve embedding_id -> criminal_id and details via DB join
                all_results.append(SearchResult(criminal_id=embedding_id, criminal_name="Unknown", similarity=similarity, threat_level=None, crime_type=None))

    # Unique by criminal_id, higher similarity wins
    best = {}
    for r in all_results:
        if r.criminal_id not in best or r.similarity > best[r.criminal_id].similarity:
            best[r.criminal_id] = r

    final = sorted(best.values(), key=lambda x: x.similarity, reverse=True)[:max_results]
    return SearchResponse(results=final, query_time_ms=0.0)
