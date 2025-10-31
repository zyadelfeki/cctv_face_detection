from typing import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from loguru import logger

from .models import Base


class Database:
    def __init__(self, config):
        self.config = config
        db_config = config.get().database
        
        # Build sync URL for migrations
        self.sync_url = f"postgresql://{db_config.user}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.name}"
        
        # Build async URL for runtime
        self.async_url = f"postgresql+asyncpg://{db_config.user}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.name}"
        
        # Create engines
        self.sync_engine = create_engine(
            self.sync_url,
            pool_size=db_config.pool_size,
            max_overflow=db_config.max_overflow,
            pool_timeout=db_config.pool_timeout,
            pool_recycle=db_config.pool_recycle,
        )
        
        self.async_engine = create_async_engine(
            self.async_url,
            pool_size=db_config.pool_size,
            max_overflow=db_config.max_overflow,
            pool_timeout=db_config.pool_timeout,
            pool_recycle=db_config.pool_recycle,
        )
        
        # Create session factories
        self.SessionLocal = sessionmaker(
            bind=self.sync_engine,
            autocommit=False,
            autoflush=False
        )
        
        self.AsyncSessionLocal = sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False
        )
        
        self.initialized = False

    async def initialize(self):
        """Initialize database connection and verify connectivity."""
        logger.info("Initializing database connection...")
        try:
            # Test async connection
            async with self.async_engine.begin() as conn:
                await conn.execute("SELECT 1")
            self.initialized = True
            logger.success("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    async def create_tables(self):
        """Create all tables (for development; use Alembic in production)."""
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.success("Database tables created")

    async def setup(self):
        """Setup database: initialize connection and create tables."""
        await self.initialize()
        await self.create_tables()
        logger.success("Database setup completed")

    async def test_connection(self):
        """Test database connectivity."""
        async with self.async_engine.begin() as conn:
            result = await conn.execute("SELECT version()")
            version = result.scalar()
            logger.info(f"Connected to PostgreSQL: {version}")
        return True

    async def close(self):
        """Close database connections."""
        await self.async_engine.dispose()
        self.sync_engine.dispose()
        logger.info("Database connections closed")

    def get_sync_session(self):
        """Get synchronous session."""
        return self.SessionLocal()

    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get asynchronous session."""
        async with self.AsyncSessionLocal() as session:
            try:
                yield session
            finally:
                await session.close()
