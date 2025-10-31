#!/usr/bin/env python3
import asyncio
import json
from pathlib import Path

import numpy as np
from loguru import logger

from src.utils.config import Config
from src.utils.logger import setup_logging
from src.database.session import Database
from src.database.services import CriminalService
from src.database.embedding_index import EmbeddingIndex


async def rebuild_faiss_index():
    """Rebuild FAISS index from database embeddings."""
    config = Config()
    setup_logging(config)
    
    logger.info("Starting FAISS index rebuild...")
    
    # Initialize database
    database = Database(config)
    await database.initialize()
    
    # Create new empty index
    embedding_index = EmbeddingIndex(dim=512, metric="cosine", persist_path="./data/faiss.index")
    
    async with database.AsyncSessionLocal() as session:
        # Get all embeddings from database
        from sqlalchemy import select
        from src.database.models import Embedding
        
        query = select(Embedding)
        result = await session.execute(query)
        embeddings = result.scalars().all()
        
        if not embeddings:
            logger.warning("No embeddings found in database")
            await database.close()
            return
        
        logger.info(f"Found {len(embeddings)} embeddings to rebuild")
        
        # Prepare vectors and IDs
        vectors = []
        ids = []
        
        for emb in embeddings:
            try:
                # Parse vector from JSON string
                vector = np.array(json.loads(emb.vector), dtype=np.float32)
                vectors.append(vector)
                ids.append(emb.id)
            except Exception as e:
                logger.warning(f"Failed to parse embedding {emb.id}: {e}")
        
        if vectors:
            # Add all vectors to index
            vectors_array = np.vstack(vectors)
            embedding_index.add(vectors_array, ids)
            
            # Save index
            embedding_index.save()
            
            logger.success(f"Successfully rebuilt FAISS index with {len(vectors)} embeddings")
        else:
            logger.error("No valid embeddings found to rebuild index")
    
    await database.close()


if __name__ == "__main__":
    asyncio.run(rebuild_faiss_index())
