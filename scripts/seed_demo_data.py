#!/usr/bin/env python3
import asyncio
import json
from pathlib import Path

import numpy as np
from loguru import logger

from src.utils.config import Config
from src.utils.logger import setup_logging
from src.database.session import Database
from src.database.services import CriminalService, CameraService
from src.database.embedding_index import EmbeddingIndex


# Demo data
DEMO_CRIMINALS = [
    {
        "name": "John Doe",
        "age": 35,
        "gender": "Male",
        "crime_type": "Armed Robbery",
        "threat_level": "High",
        "external_id": "CR001",
        "notes": "Armed and dangerous. Last seen in downtown area."
    },
    {
        "name": "Jane Smith",
        "age": 28,
        "gender": "Female",
        "crime_type": "Fraud",
        "threat_level": "Medium",
        "external_id": "CR002",
        "notes": "Known for identity theft and credit card fraud."
    },
    {
        "name": "Mike Johnson",
        "age": 42,
        "gender": "Male",
        "crime_type": "Drug Trafficking",
        "threat_level": "High",
        "external_id": "CR003",
        "notes": "Leader of drug trafficking ring."
    }
]

DEMO_CAMERAS = [
    {
        "name": "Main Entrance",
        "url": "rtsp://demo:demo@192.168.1.100/stream1",
        "location": "Building Main Entrance",
        "active": True
    },
    {
        "name": "Parking Lot",
        "url": "rtsp://demo:demo@192.168.1.101/stream1",
        "location": "Parking Area - North Side",
        "active": True
    },
    {
        "name": "Emergency Exit",
        "url": "rtsp://demo:demo@192.168.1.102/stream1",
        "location": "Emergency Exit - East Wing",
        "active": False
    }
]


async def seed_criminals(criminal_service: CriminalService):
    """Seed demo criminal records."""
    logger.info("Seeding demo criminal records...")
    
    for criminal_data in DEMO_CRIMINALS:
        try:
            # Create dummy embeddings (in production, these would come from actual photos)
            dummy_embeddings = [
                np.random.rand(512).astype(np.float32),  # Primary photo embedding
                np.random.rand(512).astype(np.float32),  # Secondary angle embedding
            ]
            
            criminal = await criminal_service.create_criminal(
                embeddings=dummy_embeddings,
                **criminal_data
            )
            
            logger.success(f"Created criminal: {criminal.name} (ID: {criminal.id})")
            
        except Exception as e:
            logger.error(f"Failed to create criminal {criminal_data['name']}: {e}")


async def seed_cameras(camera_service: CameraService):
    """Seed demo camera records."""
    logger.info("Seeding demo camera records...")
    
    for camera_data in DEMO_CAMERAS:
        try:
            camera = await camera_service.create_camera(**camera_data)
            logger.success(f"Created camera: {camera.name} (ID: {camera.id})")
        except Exception as e:
            logger.error(f"Failed to create camera {camera_data['name']}: {e}")


async def main():
    """Seed the database with demo data."""
    config = Config()
    setup_logging(config)
    
    logger.info("Starting database seeding with demo data...")
    
    # Initialize database and services
    database = Database(config)
    await database.initialize()
    
    # Initialize embedding index
    embedding_index = EmbeddingIndex(dim=512, metric="cosine", persist_path="./data/faiss.index")
    try:
        embedding_index.load()
    except Exception:
        logger.info("Starting with empty FAISS index")
    
    # Create services
    async with database.AsyncSessionLocal() as session:
        criminal_service = CriminalService(session, embedding_index)
        camera_service = CameraService(session)
        
        # Seed data
        await seed_criminals(criminal_service)
        await seed_cameras(camera_service)
    
    # Close database connection
    await database.close()
    
    logger.success("Demo data seeding completed!")
    logger.info("You can now test the API endpoints with the demo data")


if __name__ == "__main__":
    asyncio.run(main())
