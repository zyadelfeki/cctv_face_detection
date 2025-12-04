"""
Real-time video streaming service for the dashboard.
Handles RTSP/camera feeds and streams processed frames.
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import Dict, Optional, Callable, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import base64
import logging

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Container for processed frame data."""
    frame: np.ndarray
    timestamp: datetime
    camera_id: str
    faces: List[Dict]  # List of detected faces with bboxes and info
    fps: float
    

class CameraStream:
    """Handles a single camera stream."""
    
    def __init__(
        self,
        camera_id: str,
        source: str,  # RTSP URL, file path, or device index
        frame_callback: Optional[Callable] = None,
        target_fps: int = 30,
        buffer_size: int = 5
    ):
        self.camera_id = camera_id
        self.source = source
        self.frame_callback = frame_callback
        self.target_fps = target_fps
        
        self.frame_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None
        
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.status = "disconnected"
        self.last_frame: Optional[np.ndarray] = None
        self.error_count = 0
        self.max_errors = 10
    
    def start(self) -> bool:
        """Start the camera stream."""
        try:
            # Parse source
            if isinstance(self.source, int) or self.source.isdigit():
                self.cap = cv2.VideoCapture(int(self.source))
            else:
                self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}: {self.source}")
                self.status = "error"
                return False
            
            # Configure capture
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            self.running = True
            self.status = "online"
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            
            logger.info(f"Camera {self.camera_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera {self.camera_id}: {e}")
            self.status = "error"
            return False
    
    def stop(self):
        """Stop the camera stream."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        self.status = "disconnected"
        logger.info(f"Camera {self.camera_id} stopped")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        frame_interval = 1.0 / self.target_fps
        
        while self.running:
            start_time = time.time()
            
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    self.error_count += 1
                    if self.error_count >= self.max_errors:
                        logger.error(f"Camera {self.camera_id}: Too many errors, stopping")
                        self.status = "error"
                        break
                    time.sleep(0.1)
                    continue
                
                self.error_count = 0
                self.last_frame = frame
                self.frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = current_time
                
                # Put frame in queue (non-blocking)
                frame_data = FrameData(
                    frame=frame,
                    timestamp=datetime.now(),
                    camera_id=self.camera_id,
                    faces=[],
                    fps=self.fps
                )
                
                try:
                    self.frame_queue.put_nowait(frame_data)
                except queue.Full:
                    # Drop oldest frame
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame_data)
                    except queue.Empty:
                        pass
                
                # Call callback if provided
                if self.frame_callback:
                    self.frame_callback(frame_data)
                
            except Exception as e:
                logger.error(f"Error in capture loop for {self.camera_id}: {e}")
                self.error_count += 1
            
            # Maintain target FPS
            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
        
        self.status = "disconnected"
    
    def get_frame(self) -> Optional[FrameData]:
        """Get the latest frame from the queue."""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame (may skip frames)."""
        return self.last_frame


class StreamManager:
    """Manages multiple camera streams."""
    
    def __init__(self):
        self.streams: Dict[str, CameraStream] = {}
        self.detection_callback: Optional[Callable] = None
        self._lock = threading.Lock()
    
    def add_camera(
        self,
        camera_id: str,
        source: str,
        **kwargs
    ) -> bool:
        """Add a new camera stream."""
        with self._lock:
            if camera_id in self.streams:
                logger.warning(f"Camera {camera_id} already exists")
                return False
            
            stream = CameraStream(
                camera_id=camera_id,
                source=source,
                frame_callback=self._on_frame,
                **kwargs
            )
            
            if stream.start():
                self.streams[camera_id] = stream
                return True
            return False
    
    def remove_camera(self, camera_id: str) -> bool:
        """Remove a camera stream."""
        with self._lock:
            if camera_id not in self.streams:
                return False
            
            self.streams[camera_id].stop()
            del self.streams[camera_id]
            return True
    
    def get_camera_status(self, camera_id: str) -> Optional[Dict]:
        """Get status of a specific camera."""
        if camera_id not in self.streams:
            return None
        
        stream = self.streams[camera_id]
        return {
            "camera_id": camera_id,
            "status": stream.status,
            "fps": stream.fps,
            "source": stream.source
        }
    
    def get_all_status(self) -> Dict[str, Dict]:
        """Get status of all cameras."""
        return {
            cam_id: self.get_camera_status(cam_id)
            for cam_id in self.streams
        }
    
    def get_frame(self, camera_id: str) -> Optional[FrameData]:
        """Get frame from a specific camera."""
        if camera_id not in self.streams:
            return None
        return self.streams[camera_id].get_frame()
    
    def get_latest_frames(self) -> Dict[str, np.ndarray]:
        """Get latest frame from all cameras."""
        frames = {}
        for cam_id, stream in self.streams.items():
            frame = stream.get_latest_frame()
            if frame is not None:
                frames[cam_id] = frame
        return frames
    
    def _on_frame(self, frame_data: FrameData):
        """Called when a new frame is captured."""
        if self.detection_callback:
            self.detection_callback(frame_data)
    
    def set_detection_callback(self, callback: Callable):
        """Set callback for frame processing."""
        self.detection_callback = callback
    
    def stop_all(self):
        """Stop all camera streams."""
        with self._lock:
            for stream in self.streams.values():
                stream.stop()
            self.streams.clear()


def frame_to_base64(frame: np.ndarray, quality: int = 80) -> str:
    """Convert frame to base64 encoded JPEG."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_param)
    return base64.b64encode(buffer).decode('utf-8')


def draw_detections(
    frame: np.ndarray,
    faces: List[Dict],
    show_names: bool = True,
    show_confidence: bool = True
) -> np.ndarray:
    """Draw face detection boxes and labels on frame."""
    frame_copy = frame.copy()
    
    for face in faces:
        bbox = face.get('bbox', [])
        if len(bbox) != 4:
            continue
        
        x, y, w, h = map(int, bbox)
        
        # Color based on known/unknown
        is_known = face.get('is_known', False)
        color = (0, 255, 0) if is_known else (0, 165, 255)  # Green for known, orange for unknown
        
        # Draw bounding box
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, 2)
        
        # Draw label background
        label_parts = []
        if show_names:
            name = face.get('name', 'Unknown')
            label_parts.append(name)
        if show_confidence:
            conf = face.get('confidence', 0)
            label_parts.append(f"{conf:.0%}")
        
        if label_parts:
            label = " | ".join(label_parts)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                frame_copy,
                (x, y - label_size[1] - 10),
                (x + label_size[0] + 10, y),
                color,
                -1
            )
            cv2.putText(
                frame_copy,
                label,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
    
    return frame_copy


def add_overlay(
    frame: np.ndarray,
    camera_id: str,
    fps: float,
    timestamp: Optional[datetime] = None
) -> np.ndarray:
    """Add info overlay to frame."""
    frame_copy = frame.copy()
    h, w = frame_copy.shape[:2]
    
    # Draw semi-transparent header
    overlay = frame_copy.copy()
    cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
    frame_copy = cv2.addWeighted(overlay, 0.5, frame_copy, 0.5, 0)
    
    # Add camera ID
    cv2.putText(
        frame_copy,
        f"CAM: {camera_id}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )
    
    # Add FPS
    cv2.putText(
        frame_copy,
        f"FPS: {fps:.1f}",
        (w - 100, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )
    
    # Add timestamp
    if timestamp:
        time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame_copy,
            time_str,
            (w // 2 - 80, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return frame_copy
