#!/usr/bin/env python3
"""
CCTV Face Detection System - Main Entry Point

A comprehensive real-time face detection and recognition system for criminal identification.
Authors: Zyad El Feki and team
Version: 1.0.0
"""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional

import click
from loguru import logger

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.detection_engine import DetectionEngine
from src.utils.config import Config
from src.utils.logger import setup_logging
from src.database.session import Database
from src.api.main import create_app


class CCTVSystem:
    """Main CCTV Face Detection System coordinator."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = Config(config_path)
        self.detection_engine: Optional[DetectionEngine] = None
        self.database: Optional[Database] = None
        self.running = False
        
        # Setup logging
        setup_logging(self.config)
        logger.info("CCTV Face Detection System initializing...")
        
    async def initialize(self):
        """Initialize all system components."""
        try:
            # Initialize database
            logger.info("Initializing database connection...")
            self.database = Database(self.config)
            await self.database.initialize()
            
            # Initialize detection engine
            logger.info("Initializing detection engine...")
            self.detection_engine = DetectionEngine(self.config, self.database)
            await self.detection_engine.initialize()
            
            logger.success("System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    async def start(self):
        """Start the CCTV monitoring system."""
        if not await self.initialize():
            logger.error("System initialization failed. Exiting.")
            return False
            
        self.running = True
        logger.info("Starting CCTV Face Detection System...")
        
        try:
            # Start detection engine
            await self.detection_engine.start()
            
            # Keep running until stopped
            while self.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            await self.stop()
            
        return True
    
    async def stop(self):
        """Stop the CCTV monitoring system."""
        logger.info("Stopping CCTV Face Detection System...")
        self.running = False
        
        if self.detection_engine:
            await self.detection_engine.stop()
            
        if self.database:
            await self.database.close()
            
        logger.success("System stopped successfully")
    
    def handle_signal(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False


@click.group()
@click.option('--config', '-c', default=None, help='Path to configuration file')
@click.option('--debug', '-d', is_flag=True, help='Enable debug mode')
@click.pass_context
def cli(ctx, config, debug):
    """CCTV Face Detection System - Criminal Identification Platform."""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['debug'] = debug


@cli.command()
@click.pass_context
def start(ctx):
    """Start the CCTV face detection system."""
    config_path = ctx.obj.get('config_path')
    
    # Create and configure system
    system = CCTVSystem(config_path)
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, system.handle_signal)
    signal.signal(signal.SIGTERM, system.handle_signal)
    
    # Start the system
    asyncio.run(system.start())


@cli.command()
@click.option('--host', default='0.0.0.0', help='API server host')
@click.option('--port', default=8000, help='API server port')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
@click.pass_context
def api(ctx, host, port, reload):
    """Start the REST API server."""
    config_path = ctx.obj.get('config_path')
    config = Config(config_path)
    
    # Setup logging
    setup_logging(config)
    
    # Create FastAPI app
    app = create_app(config)
    
    # Start server
    import uvicorn
    uvicorn.run(
        "src.api.main:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


@cli.command()
@click.pass_context
def setup(ctx):
    """Setup database and initialize system."""
    config_path = ctx.obj.get('config_path')
    
    async def setup_system():
        config = Config(config_path)
        setup_logging(config)
        
        logger.info("Setting up CCTV Face Detection System...")
        
        # Initialize database
        database = Database(config)
        await database.setup()
        
        logger.success("System setup completed successfully")
    
    asyncio.run(setup_system())


@cli.command()
@click.pass_context
def test(ctx):
    """Run system tests and validation."""
    config_path = ctx.obj.get('config_path')
    
    async def run_tests():
        config = Config(config_path)
        setup_logging(config)
        
        logger.info("Running system tests...")
        
        # Test database connection
        try:
            database = Database(config)
            await database.test_connection()
            logger.success("✓ Database connection test passed")
        except Exception as e:
            logger.error(f"✗ Database connection test failed: {e}")
        
        # Test detection engine
        try:
            detection_engine = DetectionEngine(config, None)
            await detection_engine.test_models()
            logger.success("✓ Detection models test passed")
        except Exception as e:
            logger.error(f"✗ Detection models test failed: {e}")
        
        logger.info("System tests completed")
    
    asyncio.run(run_tests())


@cli.command()
@click.argument('image_path')
@click.pass_context
def detect(ctx, image_path):
    """Test face detection on a single image."""
    config_path = ctx.obj.get('config_path')
    
    async def detect_faces():
        config = Config(config_path)
        setup_logging(config)
        
        logger.info(f"Testing face detection on: {image_path}")
        
        detection_engine = DetectionEngine(config, None)
        await detection_engine.initialize()
        
        # Process single image
        result = await detection_engine.process_image(image_path)
        
        if result['faces']:
            logger.success(f"Detected {len(result['faces'])} face(s)")
            for i, face in enumerate(result['faces']):
                logger.info(f"Face {i+1}: confidence={face['confidence']:.3f}")
        else:
            logger.warning("No faces detected")
    
    asyncio.run(detect_faces())


if __name__ == '__main__':
    cli()