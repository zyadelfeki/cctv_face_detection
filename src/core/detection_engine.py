from __future__ import annotations
import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from loguru import logger
from prometheus_client import Counter, Gauge, Histogram

from .video_stream import VideoStream
from .pipeline import FacePipeline
from .matching import Matcher
from ..alerts.service import AlertService
from ..database.session import Database
from ..database.services import IncidentService, CriminalService
from ..database.embedding_index import EmbeddingIndex
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ..database.models import Embedding, Criminal, Camera


# Prometheus metrics
FRAMES_PROCESSED = Counter("frames_processed_total", "Total frames processed")
FACES_DETECTED = Counter("faces_detected_total", "Total faces detected")
MATCHES_FOUND = Counter("matches_total", "Total face matches found")
ALERTS_SENT = Counter("alerts_sent_total", "Total alerts sent")
ACTIVE_CAMERAS = Gauge("engine_active_cameras", "Number of active cameras")
PROCESSING_TIME = Histogram("frame_processing_seconds", "Time to process a frame")
BATCH_SIZE = Histogram("batch_size", "Number of faces per batch")


class DetectionEngine:
    """Main detection engine with GPU batching and async processing optimizations."""
    
    def __init__(self, config, database: Optional[Database] = None):
        self.config = config
        self.db = database
        self.initialized = False
        self.running = False
        self.pipeline: Optional[FacePipeline] = None
        self.matcher: Optional[Matcher] = None
        self.alerts: Optional[AlertService] = None
        self.streams: list[VideoStream] = []
        self.frame_skip = config.get().performance.frame_skip
        # Thread pool for CPU-bound operations
        self._executor = ThreadPoolExecutor(max_workers=config.get().performance.num_workers)
        # Frame queue for batch processing
        self._frame_batch: List[tuple] = []
        self._batch_size = config.get().performance.batch_size
        self._batch_timeout = config.get().performance.batch_timeout

    async def initialize(self):
        logger.info("Initializing detection engine (models, matcher, streams, alerts, database)...")
        self.pipeline = FacePipeline(self.config)
        self.matcher = Matcher(self.config)
        self.alerts = AlertService(self.config)
        # Database if not provided
        if self.db is None:
            self.db = Database(self.config)
            await self.db.initialize()
        # Initialize streams
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
        ACTIVE_CAMERAS.set(len(self.streams))
        self.initialized = True

    async def start(self):
        if not self.initialized:
            await self.initialize()
        self.running = True
        logger.info("Detection engine started")
        await asyncio.gather(*[self._run_camera(i, s) for i, s in enumerate(self.streams)])

    async def _resolve_camera_id(self, session: AsyncSession, cam_index: int) -> int:
        # Try lookup by name from config; else create ephemeral
        cfg_cam = self.config.get().cameras.sources[cam_index]
        name = cfg_cam.name if hasattr(cfg_cam, 'name') else f"Camera_{cam_index}"
        result = await session.execute(select(Camera).where(Camera.name == name))
        cam = result.scalar_one_or_none()
        if cam:
            return cam.id
        # Create on-the-fly camera record
        from ..database.services import CameraService
        cs = CameraService(session)
        cam_obj = await cs.create_camera(name=name, url=cfg_cam.url, location=getattr(cfg_cam, 'location', None), active=True)
        return cam_obj.id

    async def _resolve_matches(self, session: AsyncSession, embedding_ids: list[int]) -> Dict[int, Criminal]:
        if not embedding_ids:
            return {}
        result = await session.execute(
            select(Embedding, Criminal).join(Criminal, Embedding.criminal_id == Criminal.id).where(Embedding.id.in_(embedding_ids))
        )
        mapping: Dict[int, Criminal] = {}
        for emb, crim in result.all():
            mapping[emb.id] = crim
        return mapping

    async def _run_camera(self, cam_index: int, stream: VideoStream):
        frame_count = 0
        stream.open()
        while self.running:
            frame = stream.read()
            if frame is None:
                await asyncio.sleep(0.2)
                continue
            frame_count += 1
            FRAMES_PROCESSED.inc()
            if self.frame_skip > 1 and frame_count % self.frame_skip != 0:
                continue

            result = self.pipeline.process_frame(frame)
            faces = result.get("faces", [])
            face_count = len(faces)
            if face_count:
                FACES_DETECTED.inc(face_count)
                # Prepare embeddings for matching
                embs = np.array([f["embedding"] for f in faces], dtype=np.float32)
                matches = self.matcher.match(embs)
                # Open DB session to persist incident and matches
                async with self.db.AsyncSessionLocal() as session:
                    cam_id = await self._resolve_camera_id(session, cam_index)
                    inc_service = IncidentService(session)
                    incident = await inc_service.create_incident(camera_id=cam_id, face_count=face_count)
                    # Snapshot
                    snapshot_path = None
                    try:
                        snap_dir = Path(self.config.get().storage.detected_faces_dir)
                        snap_dir.mkdir(parents=True, exist_ok=True)
                        snapshot_path = str(snap_dir / f"incident_{incident.id}_{int(time.time())}.jpg")
                        cv2.imwrite(snapshot_path, frame)
                        incident.image_path = snapshot_path
                    except Exception:
                        pass
                    # Resolve embedding IDs to criminals
                    all_ids = [m["embedding_id"] for row in matches for m in row]
                    id_to_crim = await self._resolve_matches(session, all_ids)
                    # Persist matches and send alerts
                    for face, row in zip(faces, matches):
                        for m in row:
                            MATCHES_FOUND.inc()
                            emb_id = m["embedding_id"]
                            sim = m["similarity"]
                            x1, y1, x2, y2 = face["box"]
                            # Persist
                            crim = id_to_crim.get(emb_id)
                            if crim:
                                await inc_service.add_match(incident_id=incident.id, criminal_id=crim.id, similarity=sim, bbox=(x1, y1, x2, y2))
                                # Alert with enriched info
                                self.alerts.send_alert(
                                    criminal_name=crim.name,
                                    similarity=sim,
                                    camera_name=self.config.get().cameras.sources[cam_index].name,
                                    location=getattr(self.config.get().cameras.sources[cam_index], 'location', None),
                                    snapshot_path=snapshot_path,
                                )
                                ALERTS_SENT.inc()
                    # Commit
                    await session.commit()
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
