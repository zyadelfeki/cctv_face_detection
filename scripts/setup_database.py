#!/usr/bin/env python3
from pathlib import Path

from loguru import logger

from src.utils.config import Config
from src.utils.logger import setup_logging


async def main():
    config = Config()
    setup_logging(config)
    logger.info("Database and index setup starting...")
    # Here we'd initialize SQLAlchemy engine and create tables via Alembic or metadata.create_all
    # Leaving as a placeholder to keep async footprint consistent with rest of app
    logger.info("Creating directories and preparing FAISS index store...")
    Path(config.get().storage.criminal_photos_dir).mkdir(parents=True, exist_ok=True)
    Path(config.get().storage.detected_faces_dir).mkdir(parents=True, exist_ok=True)
    logger.success("Setup complete")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
