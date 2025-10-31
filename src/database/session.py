import asyncio
from typing import Optional

from loguru import logger


class Database:
    def __init__(self, config):
        self.config = config
        self.initialized = False

    async def initialize(self):
        logger.info("Initializing database (stub)...")
        await asyncio.sleep(0.1)
        self.initialized = True

    async def setup(self):
        logger.info("Setting up database schema (stub)...")
        await asyncio.sleep(0.1)

    async def test_connection(self):
        await asyncio.sleep(0.05)
        return True

    async def close(self):
        logger.info("Closing database (stub)")
        await asyncio.sleep(0.05)
