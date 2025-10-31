from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from ..database.session import Database
from ..database.embedding_index import EmbeddingIndex
from ..core.pipeline import FacePipeline
from ..utils.logger import setup_logging

from .criminals import router as criminals_router
from .incidents import router as incidents_router
from .cameras import router as cameras_router


# Global instances
database: Optional[Database] = None
embedding_index: Optional[EmbeddingIndex] = None
face_pipeline: Optional[FacePipeline] = None


def create_app(config) -> FastAPI:
    """Create FastAPI application with all routes and middleware."""
    app_config = config.get()
    
    # Initialize FastAPI
    app = FastAPI(
        title=app_config.api.title,
        description=app_config.api.description,
        version=app_config.api.version
    )
    
    # Setup logging
    setup_logging(config)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_config.web.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Internal server error", "detail": str(exc)}
        )
    
    # Dependency injection setup
    async def get_database():
        global database
        if database is None:
            database = Database(config)
            await database.initialize()
        return database
    
    async def get_db_session():
        db = await get_database()
        async with db.AsyncSessionLocal() as session:
            try:
                yield session
            finally:
                await session.close()
    
    async def get_embedding_index():
        global embedding_index
        if embedding_index is None:
            embedding_index = EmbeddingIndex(dim=512, metric="cosine", persist_path="./data/faiss.index")
            try:
                embedding_index.load()
            except Exception:
                pass  # Start with empty index
        return embedding_index
    
    async def get_face_pipeline():
        global face_pipeline
        if face_pipeline is None:
            face_pipeline = FacePipeline(config)
        return face_pipeline
    
    # Override the dependency functions in routers
    criminals_router.get_db_session = get_db_session
    incidents_router.get_db_session = get_db_session
    cameras_router.get_db_session = get_db_session
    criminals_router.get_face_pipeline = get_face_pipeline
    
    # Health endpoints
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok", "timestamp": time.time()}
    
    @app.get("/version")
    async def version():
        """Version endpoint."""
        return {"version": app_config.api.version}
    
    @app.get("/metrics")
    async def metrics():
        """Basic metrics endpoint."""
        # In production, this would return Prometheus metrics
        return {
            "database_connected": database.initialized if database else False,
            "embedding_index_size": len(embedding_index.ids) if embedding_index else 0,
        }
    
    # Include routers
    app.include_router(criminals_router)
    app.include_router(incidents_router)
    app.include_router(cameras_router)
    
    return app
