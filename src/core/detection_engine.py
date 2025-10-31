import asyncio
from typing import Any, Dict, Optional

import cv2
from loguru import logger

from .video_stream import VideoStream
from .pipeline import FacePipeline


class DetectionEngine:
    def __init__(self, config, database=None):
        self.config = config
        self.db = database
        self.initialized = False
        self.running = False
        self.pipeline: Optional[FacePipeline] = None
        self.streams: list[VideoStream] = []
        self.frame_skip = config.get().performance.frame_skip

    async def initialize(self):
        logger.info("Initializing detection engine (models and streams)...")
        self.pipeline = FacePipeline(self.config)
        # Initialize streams from config
        self.streams = []
        for cam in self.config.get().cameras.sources:
            if cam.active:
                self.streams.append(
                    VideoStream(
                        cam.url,
                        buffer_size=self.config.get().cameras.buffer_size,
                        reconnect_interval=self.config.get().cameras.reconnect_interval,
                        timeout=self.config.get().cameras.timeout,
                    )
                )
        self.initialized = True

    async def start(self):
        if not self.initialized:
            await self.initialize()
        self.running = True
        logger.info("Detection engine started")
        await asyncio.gather(*[self._run_camera(i, s) for i, s in enumerate(self.streams)])

    async def _run_camera(self, cam_index: int, stream: VideoStream):
        frame_count = 0
        stream.open()
        while self.running:
            frame = stream.read()
            if frame is None:
                await asyncio.sleep(0.2)
                continue
            frame_count += 1
            if self.frame_skip > 1 and frame_count % self.frame_skip != 0:
                continue
            result = self.pipeline.process_frame(frame)
            if result["count"]:
                # TODO: match embeddings to DB, emit alerts, log incidents
                logger.info(f"Cam {cam_index}: detected {result['count']} face(s)")
            await asyncio.sleep(0)
        stream.close()

    async def stop(self):
        self.running = False
        logger.info("Detection engine stopped")

    async def test_models(self):
        await self.initialize()
        return True

    async def process_image(self, image_path: str) -> Dict[str, Any]:
        img = cv2.imread(image_path)
        if img is None:
            return {"image": image_path, "faces": []}
        return self.pipeline.process_frame(img)
