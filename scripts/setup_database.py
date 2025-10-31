#!/usr/bin/env python3
import asyncio
from pathlib import Path

from loguru import logger

from src.utils.config import Config
from src.utils.logger import setup_logging
from src.database.session import Database
from src.database.embedding_index import EmbeddingIndex


async def main():
    """Setup database schema and initialize FAISS index."""
    config = Config()
    setup_logging(config)
    
    logger.info("Starting database and index setup...")
    
    # Create storage directories
    storage_config = config.get().storage
    directories = [
        storage_config.criminal_photos_dir,
        storage_config.detected_faces_dir,
        storage_config.logs_dir,
        storage_config.temp_dir,
        storage_config.backup_dir,
        "./data"  # For FAISS index
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Initialize database
    try:
        database = Database(config)
        await database.setup()
        logger.success("Database setup completed")
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise
    
    # Initialize FAISS index
    try:
        embedding_index = EmbeddingIndex(dim=512, metric="cosine", persist_path="./data/faiss.index")
        embedding_index.save()  # Create empty index file
        logger.success("FAISS index initialized")
    except Exception as e:
        logger.error(f"FAISS index setup failed: {e}")
        raise
    
    # Close database connection
    await database.close()
    
    logger.success("Setup completed successfully!")
    logger.info("You can now run: python main.py start")


if __name__ == "__main__":
    asyncio.run(main())
