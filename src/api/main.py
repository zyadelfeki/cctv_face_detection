from typing import Optional

from fastapi import FastAPI


def create_app(config) -> FastAPI:
    app = FastAPI(title=config.get().api.title, description=config.get().api.description, version=config.get().api.version)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/version")
    async def version():
        return {"version": config.get().api.version}

    return app
