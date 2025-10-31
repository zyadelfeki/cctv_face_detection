from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Depends
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
from .auth import AuthService
from .auth_routes import create_auth_router
from .rate_limit import use_rate_limit


# Global instances
database: Optional[Database] = None
embedding_index: Optional[EmbeddingIndex] = None
face_pipeline: Optional[FacePipeline] = None
auth_service: Optional[AuthService] = None


def create_app(config) -> FastAPI:
    """Create FastAPI application with all routes, middleware, auth and rate limiting."""
    app_config = config.get()
    
    app = FastAPI(
        title=app_config.api.title,
        description=app_config.api.description,
        version=app_config.api.version
    )
    
    setup_logging(config)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_config.web.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize auth
    global auth_service
    auth_service = AuthService(
        secret_key=app_config.api.authentication.secret_key,
        algorithm=app_config.api.authentication.algorithm,
        access_token_expire_minutes=app_config.api.authentication.access_token_expire_minutes
    )
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"success": False, "error": "Internal server error", "detail": str(exc)})
    
    # Dependencies
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
                pass
        return embedding_index
    
    async def get_face_pipeline():
        global face_pipeline
        if face_pipeline is None:
            face_pipeline = FacePipeline(config)
        return face_pipeline
    
    # Patch routers with dependencies
    criminals_router.get_db_session = get_db_session
    incidents_router.get_db_session = get_db_session
    cameras_router.get_db_session = get_db_session
    criminals_router.get_face_pipeline = get_face_pipeline
    
    # Apply rate limiting globally to API routes if enabled
    if app_config.api.rate_limiting.enabled:
        limiter_dep = use_rate_limit(app_config.api.rate_limiting.calls, app_config.api.rate_limiting.period)
        app.dependency_overrides[use_rate_limit] = limiter_dep
    
    # Auth routes
    app.include_router(create_auth_router(auth_service))
    
    # Health and meta
    @app.get("/health")
    async def health():
        return {"status": "ok", "timestamp": time.time()}
    
    @app.get("/version")
    async def version():
        return {"version": app_config.api.version}
    
    @app.get("/metrics")
    async def metrics():
        return {
            "database_connected": database.initialized if database else False,
            "embedding_index_size": len(embedding_index.ids) if embedding_index else 0,
        }
    
    # Protected routers (JWT will be enforced at endpoint level in future commit)
    app.include_router(criminals_router)
    app.include_router(incidents_router)
    app.include_router(cameras_router)
    
    return app
