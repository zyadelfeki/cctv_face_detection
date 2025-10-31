from time import time
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CollectorRegistry, CONTENT_TYPE_LATEST, generate_latest, Histogram

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

# Global registry and histogram
REGISTRY = CollectorRegistry()
API_LATENCY = Histogram(
    "api_request_duration_seconds",
    "API request latency",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)

# Globals
_database = None
_embedding_index = None
_face_pipeline = None
_auth_service = None


def create_app(config) -> FastAPI:
    app_config = config.get()

    app = FastAPI(title=app_config.api.title, description=app_config.api.description, version=app_config.api.version)
    setup_logging(config)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_config.web.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware for latency
    @app.middleware("http")
    async def add_latency_metrics(request: Request, call_next: Callable):
        start = time()
        response = await call_next(request)
        duration = time() - start
        API_LATENCY.observe(duration)
        return response

    # Initialize auth
    global _auth_service
    _auth_service = AuthService(
        secret_key=app_config.api.authentication.secret_key,
        algorithm=app_config.api.authentication.algorithm,
        access_token_expire_minutes=app_config.api.authentication.access_token_expire_minutes,
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"success": False, "error": "Internal server error", "detail": str(exc)})

    async def get_database():
        global _database
        if _database is None:
            _database = Database(config)
            await _database.initialize()
        return _database

    async def get_db_session():
        db = await get_database()
        async with db.AsyncSessionLocal() as session:
            try:
                yield session
            finally:
                await session.close()

    async def get_embedding_index():
        global _embedding_index
        if _embedding_index is None:
            _embedding_index = EmbeddingIndex(dim=512, metric="cosine", persist_path="./data/faiss.index")
            try:
                _embedding_index.load()
            except Exception:
                pass
        return _embedding_index

    async def get_face_pipeline():
        global _face_pipeline
        if _face_pipeline is None:
            _face_pipeline = FacePipeline(config)
        return _face_pipeline

    # Inject dependencies
    criminals_router.get_db_session = get_db_session
    incidents_router.get_db_session = get_db_session
    cameras_router.get_db_session = get_db_session
    criminals_router.get_face_pipeline = get_face_pipeline

    # Rate limit
    if app_config.api.rate_limiting.enabled:
        limiter_dep = use_rate_limit(app_config.api.rate_limiting.calls, app_config.api.rate_limiting.period)
        app.dependency_overrides[use_rate_limit] = limiter_dep

    # Auth
    app.include_router(create_auth_router(_auth_service))

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/version")
    async def version():
        return {"version": app_config.api.version}

    @app.get("/metrics")
    async def metrics():
        # prometheus_client default registry exposure
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.include_router(criminals_router)
    app.include_router(incidents_router)
    app.include_router(cameras_router)

    return app
