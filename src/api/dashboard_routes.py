"""
Dashboard API endpoints for real-time data.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
import json
import cv2
import numpy as np
from pydantic import BaseModel
from collections import deque
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


# In-memory storage for demo (use Redis/DB in production)
class DashboardStore:
    def __init__(self):
        self.detections = deque(maxlen=10000)
        self.alerts = deque(maxlen=1000)
        self.camera_stats: Dict[str, Dict] = {}
        self.connected_websockets: List[WebSocket] = []
    
    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients."""
        disconnected = []
        for ws in self.connected_websockets:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)
        
        for ws in disconnected:
            self.connected_websockets.remove(ws)


store = DashboardStore()


# Pydantic models
class DetectionEvent(BaseModel):
    timestamp: datetime
    camera_id: str
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    confidence: float
    is_known: bool
    bbox: List[int]


class AlertEvent(BaseModel):
    timestamp: datetime
    camera_id: str
    severity: str  # low, medium, high, critical
    title: str
    message: str
    snapshot_url: Optional[str] = None


class CameraStatus(BaseModel):
    camera_id: str
    status: str  # online, offline, error
    fps: float
    total_detections: int
    known_faces: int
    unknown_faces: int
    last_detection: Optional[datetime] = None


# REST Endpoints
@router.get("/stats")
async def get_dashboard_stats():
    """Get overall dashboard statistics."""
    now = datetime.now()
    hour_ago = now - timedelta(hours=1)
    
    recent_detections = [
        d for d in store.detections 
        if d.get('timestamp', datetime.min) > hour_ago
    ]
    
    total_cameras = len(store.camera_stats)
    online_cameras = sum(
        1 for s in store.camera_stats.values() 
        if s.get('status') == 'online'
    )
    
    return {
        "total_detections_hour": len(recent_detections),
        "known_faces_hour": sum(1 for d in recent_detections if d.get('is_known')),
        "unknown_faces_hour": sum(1 for d in recent_detections if not d.get('is_known')),
        "active_alerts": len(store.alerts),
        "total_cameras": total_cameras,
        "online_cameras": online_cameras,
        "timestamp": now.isoformat()
    }


@router.get("/detections")
async def get_detections(
    minutes: int = 60,
    camera_id: Optional[str] = None,
    is_known: Optional[bool] = None,
    limit: int = 100
):
    """Get recent detections with optional filters."""
    cutoff = datetime.now() - timedelta(minutes=minutes)
    
    detections = [
        d for d in store.detections
        if d.get('timestamp', datetime.min) > cutoff
    ]
    
    if camera_id:
        detections = [d for d in detections if d.get('camera_id') == camera_id]
    
    if is_known is not None:
        detections = [d for d in detections if d.get('is_known') == is_known]
    
    # Sort by timestamp descending
    detections.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
    
    return {
        "detections": detections[:limit],
        "total": len(detections),
        "filtered": len(detections) > limit
    }


@router.get("/alerts")
async def get_alerts(
    severity: Optional[str] = None,
    limit: int = 50
):
    """Get recent alerts."""
    alerts = list(store.alerts)
    
    if severity:
        alerts = [a for a in alerts if a.get('severity') == severity]
    
    return {
        "alerts": alerts[:limit],
        "total": len(alerts)
    }


@router.get("/cameras")
async def get_cameras():
    """Get all camera statuses."""
    return {
        "cameras": list(store.camera_stats.values()),
        "total": len(store.camera_stats)
    }


@router.get("/cameras/{camera_id}")
async def get_camera(camera_id: str):
    """Get specific camera status."""
    if camera_id not in store.camera_stats:
        raise HTTPException(status_code=404, detail="Camera not found")
    return store.camera_stats[camera_id]


@router.get("/hourly-stats")
async def get_hourly_stats():
    """Get detection counts by hour for the last 24 hours."""
    now = datetime.now()
    hourly = {h: {"total": 0, "known": 0, "unknown": 0} for h in range(24)}
    
    for detection in store.detections:
        ts = detection.get('timestamp')
        if ts and (now - ts).total_seconds() < 86400:  # Last 24 hours
            hour = ts.hour
            hourly[hour]["total"] += 1
            if detection.get('is_known'):
                hourly[hour]["known"] += 1
            else:
                hourly[hour]["unknown"] += 1
    
    return {"hourly_stats": hourly}


@router.get("/camera-heatmap")
async def get_camera_heatmap():
    """Get detection heatmap data by camera."""
    heatmap = {}
    
    for cam_id, stats in store.camera_stats.items():
        heatmap[cam_id] = {
            "total": stats.get('total_detections', 0),
            "known": stats.get('known_faces', 0),
            "unknown": stats.get('unknown_faces', 0),
            "alerts": stats.get('alerts_triggered', 0)
        }
    
    return {"heatmap": heatmap}


# POST endpoints for receiving data
@router.post("/detections")
async def add_detection(detection: DetectionEvent):
    """Add a new detection event."""
    detection_dict = detection.dict()
    detection_dict['timestamp'] = detection.timestamp.isoformat()
    store.detections.append(detection_dict)
    
    # Update camera stats
    cam_id = detection.camera_id
    if cam_id not in store.camera_stats:
        store.camera_stats[cam_id] = {
            'camera_id': cam_id,
            'status': 'online',
            'fps': 0,
            'total_detections': 0,
            'known_faces': 0,
            'unknown_faces': 0
        }
    
    store.camera_stats[cam_id]['total_detections'] += 1
    if detection.is_known:
        store.camera_stats[cam_id]['known_faces'] += 1
    else:
        store.camera_stats[cam_id]['unknown_faces'] += 1
    store.camera_stats[cam_id]['last_detection'] = detection.timestamp.isoformat()
    
    # Broadcast to WebSocket clients
    await store.broadcast({
        "type": "detection",
        "data": detection_dict
    })
    
    return {"status": "ok"}


@router.post("/alerts")
async def add_alert(alert: AlertEvent):
    """Add a new alert."""
    alert_dict = alert.dict()
    alert_dict['timestamp'] = alert.timestamp.isoformat()
    store.alerts.appendleft(alert_dict)
    
    # Broadcast to WebSocket clients
    await store.broadcast({
        "type": "alert",
        "data": alert_dict
    })
    
    return {"status": "ok"}


# WebSocket for real-time updates
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates."""
    await websocket.accept()
    store.connected_websockets.append(websocket)
    
    try:
        # Send initial state
        await websocket.send_json({
            "type": "init",
            "data": {
                "cameras": list(store.camera_stats.values()),
                "recent_detections": list(store.detections)[-20:],
                "alerts": list(store.alerts)[:10]
            }
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                message = json.loads(data)
                
                # Handle ping/pong
                if message.get('type') == 'ping':
                    await websocket.send_json({"type": "pong"})
                
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
                
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in store.connected_websockets:
            store.connected_websockets.remove(websocket)


# Video streaming endpoint
@router.get("/stream/{camera_id}")
async def video_stream(camera_id: str):
    """Stream video from a camera as MJPEG."""
    
    async def generate():
        # This would connect to actual camera stream
        # For demo, we'll generate placeholder frames
        while True:
            # Create a placeholder frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                f"Camera: {camera_id}",
                (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            cv2.putText(
                frame,
                datetime.now().strftime("%H:%M:%S"),
                (250, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )
            
            await asyncio.sleep(0.033)  # ~30 FPS
    
    return StreamingResponse(
        generate(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )
