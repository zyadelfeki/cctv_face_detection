#!/usr/bin/env python3
import asyncio

from loguru import logger

from src.utils.config import Config
from src.utils.logger import setup_logging
from src.database.session import Database
from src.core.pipeline import FacePipeline
from src.database.embedding_index import EmbeddingIndex


async def test_database(config: Config):
    """Test database connectivity and operations."""
    logger.info("Testing database connection...")
    
    try:
        database = Database(config)
        await database.test_connection()
        await database.close()
        logger.success("✓ Database connection test passed")
        return True
    except Exception as e:
        logger.error(f"✗ Database connection test failed: {e}")
        return False


async def test_face_pipeline(config: Config):
    """Test face detection and recognition pipeline."""
    logger.info("Testing face detection pipeline...")
    
    try:
        pipeline = FacePipeline(config)
        # Test with dummy image data
        import numpy as np
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(dummy_image)
        
        if 'faces' in result and 'count' in result:
            logger.success("✓ Face pipeline test passed")
            return True
        else:
            logger.error("✗ Face pipeline test failed: Invalid result format")
            return False
            
    except Exception as e:
        logger.error(f"✗ Face pipeline test failed: {e}")
        return False


async def test_embedding_index():
    """Test FAISS embedding index."""
    logger.info("Testing FAISS embedding index...")
    
    try:
        import numpy as np
        
        # Create test index
        index = EmbeddingIndex(dim=512, metric="cosine", persist_path="./data/test_faiss.index")
        
        # Add test vectors
        test_vectors = np.random.rand(5, 512).astype(np.float32)
        test_ids = [1, 2, 3, 4, 5]
        index.add(test_vectors, test_ids)
        
        # Test search
        query_vector = np.random.rand(1, 512).astype(np.float32)
        results = index.search(query_vector, k=3)
        
        if len(results) > 0 and len(results[0]) > 0:
            logger.success("✓ FAISS index test passed")
            return True
        else:
            logger.error("✗ FAISS index test failed: No results returned")
            return False
            
    except Exception as e:
        logger.error(f"✗ FAISS index test failed: {e}")
        return False


async def test_config_loading(config: Config):
    """Test configuration loading."""
    logger.info("Testing configuration loading...")
    
    try:
        cfg = config.get()
        
        # Check required config sections
        required_sections = ['system', 'detection', 'recognition', 'database', 'api']
        for section in required_sections:
            if not hasattr(cfg, section):
                logger.error(f"✗ Config test failed: Missing section '{section}'")
                return False
        
        logger.success("✓ Configuration loading test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration loading test failed: {e}")
        return False


async def main():
    """Run all system tests."""
    config = Config()
    setup_logging(config)
    
    logger.info("Running CCTV Face Detection System tests...")
    logger.info("=" * 50)
    
    # Run tests
    tests = [
        ("Configuration Loading", test_config_loading(config)),
        ("Database Connection", test_database(config)),
        ("FAISS Embedding Index", test_embedding_index()),
        ("Face Detection Pipeline", test_face_pipeline(config)),
    ]
    
    results = []
    for test_name, test_coro in tests:
        logger.info(f"\nRunning {test_name} test...")
        result = await test_coro
        results.append((test_name, result))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        color = "green" if result else "red"
        logger.info(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    logger.info(f"\nTests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        logger.success("All tests passed! System is ready to run.")
    else:
        logger.warning(f"{len(results) - passed} test(s) failed. Please check the configuration and dependencies.")


if __name__ == "__main__":
    asyncio.run(main())
