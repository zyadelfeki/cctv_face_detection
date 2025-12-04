"""
Mobile App API Routes

REST endpoints for mobile app integration:
- Device registration
- Push notification settings
- Adaptive video streaming
- Offline sync

Author: CCTV Face Detection System
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, WebSocket
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import numpy as np

from ..core.mobile_integration import (
    AlertNotificationService,
    DeviceManager,
    FirebaseCloudMessaging,
    MobileDevice,
    MobileStreamAdapter,
    NotificationType,
    OfflineSyncManager,
    create_mobile_notification_system
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/mobile", tags=["mobile"])

# Global instances (initialized on startup)
fcm_client: Optional[FirebaseCloudMessaging] = None
device_manager: Optional[DeviceManager] = None
alert_service: Optional[AlertNotificationService] = None
sync_manager: OfflineSyncManager = OfflineSyncManager()
stream_adapters: Dict[str, MobileStreamAdapter] = {}


# ============ Pydantic Models ============

class DeviceRegistrationRequest(BaseModel):
    """Request to register a mobile device"""
    device_id: str = Field(..., description="Unique device identifier")
    fcm_token: str = Field(..., description="Firebase Cloud Messaging token")
    device_name: str = Field(..., description="Human-readable device name")
    platform: str = Field(..., pattern="^(ios|android)$", description="Device platform")
    app_version: str = Field(..., description="App version string")
    timezone: str = Field(default="UTC", description="Device timezone")


class DeviceRegistrationResponse(BaseModel):
    """Response after device registration"""
    success: bool
    device_id: str
    message: str
    notification_settings: Dict[str, bool]


class TokenUpdateRequest(BaseModel):
    """Request to update FCM token"""
    device_id: str
    new_token: str


class NotificationSettingsRequest(BaseModel):
    """Request to update notification preferences"""
    device_id: str
    settings: Dict[str, bool]


class StreamSettingsRequest(BaseModel):
    """Request to configure video stream"""
    client_id: str
    network_type: str = Field(default="wifi", pattern="^(wifi|4g|3g|slow)$")
    screen_width: int = Field(default=720, ge=240, le=1920)


class StreamSettingsResponse(BaseModel):
    """Video stream configuration"""
    fps: int
    quality: int
    width: int


class NetworkConditionsUpdate(BaseModel):
    """Report current network conditions"""
    client_id: str
    latency_ms: float
    bandwidth_kbps: float


class SyncRequest(BaseModel):
    """Request to sync offline data"""
    device_id: str
    last_sync_timestamp: Optional[str] = None
    event_ids_to_mark_synced: List[str] = []


class SyncResponse(BaseModel):
    """Sync response with pending data"""
    sync_timestamp: str
    events: List[Dict]
    has_more: bool


class HealthCheckResponse(BaseModel):
    """Mobile service health check"""
    status: str
    firebase_connected: bool
    registered_devices: int
    active_streams: int


# ============ Lifecycle Events ============

async def init_mobile_services(
    firebase_project_id: str,
    firebase_credentials_path: Optional[str] = None
):
    """
    Initialize mobile services on app startup.
    
    Call this from your main.py startup event.
    """
    global fcm_client, device_manager, alert_service
    
    try:
        fcm_client, device_manager, alert_service = create_mobile_notification_system(
            firebase_project_id=firebase_project_id,
            firebase_credentials_path=firebase_credentials_path
        )
        await alert_service.start()
        logger.info("Mobile services initialized successfully")
    except Exception as e:
        logger.warning(f"Mobile services init failed (FCM may not be configured): {e}")
        # Create local-only services
        device_manager = DeviceManager()


async def shutdown_mobile_services():
    """Cleanup on app shutdown"""
    global alert_service
    if alert_service:
        await alert_service.stop()


# ============ Device Management ============

@router.post("/devices/register", response_model=DeviceRegistrationResponse)
async def register_device(
    request: DeviceRegistrationRequest,
    user_id: str = Query(..., description="Authenticated user ID")
):
    """
    Register a mobile device for push notifications.
    
    Call this when:
    - User logs in on a new device
    - App is freshly installed
    - FCM token is refreshed
    """
    if not device_manager:
        raise HTTPException(status_code=503, detail="Mobile services not initialized")
    
    device = device_manager.register_device(
        device_id=request.device_id,
        fcm_token=request.fcm_token,
        user_id=user_id,
        device_name=request.device_name,
        platform=request.platform,
        app_version=request.app_version,
        timezone=request.timezone
    )
    
    return DeviceRegistrationResponse(
        success=True,
        device_id=device.device_id,
        message="Device registered successfully",
        notification_settings=device.notification_settings
    )


@router.post("/devices/token")
async def update_fcm_token(request: TokenUpdateRequest):
    """
    Update FCM token for a device.
    
    FCM tokens can expire or be refreshed by Firebase.
    Call this whenever the token changes.
    """
    if not device_manager:
        raise HTTPException(status_code=503, detail="Mobile services not initialized")
    
    success = device_manager.update_fcm_token(
        device_id=request.device_id,
        new_token=request.new_token
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Device not found")
    
    return {"success": True, "message": "Token updated"}


@router.post("/devices/notifications")
async def update_notification_settings(request: NotificationSettingsRequest):
    """
    Update notification preferences for a device.
    
    Settings are per notification type. Example:
    ```json
    {
        "device_id": "abc123",
        "settings": {
            "face_detected": true,
            "unknown_person": true,
            "liveness_fail": true,
            "camera_offline": false
        }
    }
    ```
    """
    if not device_manager:
        raise HTTPException(status_code=503, detail="Mobile services not initialized")
    
    # Validate setting keys
    valid_types = {nt.value for nt in NotificationType}
    invalid_keys = set(request.settings.keys()) - valid_types
    if invalid_keys:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid notification types: {invalid_keys}"
        )
    
    success = device_manager.update_notification_settings(
        device_id=request.device_id,
        settings=request.settings
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Device not found")
    
    return {"success": True, "message": "Settings updated"}


@router.get("/devices/{device_id}")
async def get_device_info(device_id: str):
    """Get device registration info"""
    if not device_manager:
        raise HTTPException(status_code=503, detail="Mobile services not initialized")
    
    device = device_manager.devices.get(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    return {
        "device_id": device.device_id,
        "device_name": device.device_name,
        "platform": device.platform,
        "app_version": device.app_version,
        "registered_at": device.registered_at.isoformat(),
        "last_seen": device.last_seen.isoformat(),
        "is_active": device.is_active,
        "notification_settings": device.notification_settings
    }


@router.delete("/devices/{device_id}")
async def deactivate_device(device_id: str):
    """
    Deactivate a device (logout or uninstall).
    
    This stops push notifications to the device.
    """
    if not device_manager:
        raise HTTPException(status_code=503, detail="Mobile services not initialized")
    
    success = device_manager.deactivate_device(device_id)
    if not success:
        raise HTTPException(status_code=404, detail="Device not found")
    
    return {"success": True, "message": "Device deactivated"}


# ============ Video Streaming ============

@router.post("/stream/configure", response_model=StreamSettingsResponse)
async def configure_stream(request: StreamSettingsRequest):
    """
    Configure video stream for mobile client.
    
    Returns optimal settings based on network type and screen size.
    """
    adapter = stream_adapters.get(request.client_id)
    if not adapter:
        # Create new adapter (you'd pass actual frame callback here)
        adapter = MobileStreamAdapter(
            frame_callback=lambda: None,  # Placeholder
            target_fps=15,
            jpeg_quality=70
        )
        stream_adapters[request.client_id] = adapter
    
    settings = adapter.register_client(
        client_id=request.client_id,
        network_type=request.network_type,
        screen_width=request.screen_width
    )
    
    return StreamSettingsResponse(**settings)


@router.post("/stream/network")
async def update_network_conditions(request: NetworkConditionsUpdate):
    """
    Report current network conditions for adaptive streaming.
    
    Mobile app should call this periodically or when conditions change.
    """
    adapter = stream_adapters.get(request.client_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Stream not configured")
    
    settings = adapter.update_network_conditions(
        client_id=request.client_id,
        latency_ms=request.latency_ms,
        bandwidth_kbps=request.bandwidth_kbps
    )
    
    return StreamSettingsResponse(**settings)


@router.get("/stream/{camera_id}")
async def mobile_camera_stream(
    camera_id: str,
    client_id: str = Query(..., description="Registered client ID")
):
    """
    Get mobile-optimized camera stream.
    
    Returns MJPEG stream with adaptive quality based on client settings.
    """
    # This would integrate with your camera manager
    # For now, return a placeholder response
    async def generate():
        while True:
            # In real implementation: get frame from camera, encode with client settings
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + b"\xff\xd8\xff" + b"\r\n"
            await asyncio.sleep(0.1)
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ============ WebSocket for Real-time Updates ============

@router.websocket("/ws/{device_id}")
async def mobile_websocket(websocket: WebSocket, device_id: str):
    """
    WebSocket connection for real-time mobile updates.
    
    Receives:
    - Detection events
    - Camera status changes
    - System alerts
    """
    if device_manager and device_id not in device_manager.devices:
        await websocket.close(code=4001)
        return
    
    await websocket.accept()
    logger.info(f"Mobile WebSocket connected: {device_id}")
    
    try:
        while True:
            # Send heartbeat
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat()
            })
            
            # Wait for client messages or timeout
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                
                # Handle client messages
                if data.get("type") == "ack":
                    # Acknowledge received notification
                    pass
                elif data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    
            except asyncio.TimeoutError:
                continue
                
    except Exception as e:
        logger.error(f"WebSocket error for {device_id}: {e}")
    finally:
        logger.info(f"Mobile WebSocket disconnected: {device_id}")


# ============ Offline Sync ============

@router.post("/sync", response_model=SyncResponse)
async def sync_offline_data(request: SyncRequest):
    """
    Sync offline data with mobile device.
    
    Mobile app calls this when coming online to get missed events.
    """
    # Mark previously synced events
    if request.event_ids_to_mark_synced:
        sync_manager.mark_synced(
            device_id=request.device_id,
            event_ids=request.event_ids_to_mark_synced
        )
    
    # Get pending events
    since = None
    if request.last_sync_timestamp:
        since = datetime.fromisoformat(request.last_sync_timestamp)
    
    payload = sync_manager.get_sync_payload(request.device_id)
    
    return SyncResponse(
        sync_timestamp=payload["sync_timestamp"],
        events=payload["events"],
        has_more=payload["has_more"]
    )


# ============ Test Endpoints ============

@router.post("/test/notification")
async def send_test_notification(
    device_id: str = Query(..., description="Target device ID"),
    notification_type: str = Query(default="face_detected")
):
    """
    Send a test notification to verify setup.
    
    Useful for debugging push notification configuration.
    """
    if not alert_service or not device_manager:
        raise HTTPException(status_code=503, detail="Mobile services not initialized")
    
    device = device_manager.devices.get(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    # Send test notification based on type
    if notification_type == "face_detected":
        await alert_service.notify_face_detected(
            person_name="Test User",
            camera_name="Test Camera",
            confidence=0.95,
            user_ids=[device.user_id]
        )
    elif notification_type == "unknown_person":
        await alert_service.notify_unknown_person(
            camera_name="Test Camera",
            track_id="test-track-123",
            user_ids=[device.user_id]
        )
    elif notification_type == "security_alert":
        await alert_service.notify_liveness_failure(
            camera_name="Test Camera",
            failure_reason="Test liveness failure",
            user_ids=[device.user_id]
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown type: {notification_type}")
    
    return {"success": True, "message": f"Test {notification_type} notification queued"}


@router.get("/health", response_model=HealthCheckResponse)
async def mobile_health_check():
    """Health check for mobile services"""
    return HealthCheckResponse(
        status="ok",
        firebase_connected=fcm_client is not None,
        registered_devices=len(device_manager.devices) if device_manager else 0,
        active_streams=len(stream_adapters)
    )


# ============ Notification Types Reference ============

@router.get("/notification-types")
async def get_notification_types():
    """
    Get available notification types for settings UI.
    
    Returns list of notification types with descriptions.
    """
    return {
        "types": [
            {
                "key": "face_detected",
                "name": "Face Detected",
                "description": "When a known face is detected",
                "default": True
            },
            {
                "key": "unknown_person",
                "name": "Unknown Person",
                "description": "When an unrecognized face is detected",
                "default": True
            },
            {
                "key": "liveness_fail",
                "name": "Spoof Attempt",
                "description": "When liveness check fails (potential spoofing)",
                "default": True
            },
            {
                "key": "security_alert",
                "name": "Security Alert",
                "description": "General security alerts",
                "default": True
            },
            {
                "key": "person_of_interest",
                "name": "Person of Interest",
                "description": "When a flagged person is detected",
                "default": True
            },
            {
                "key": "camera_offline",
                "name": "Camera Offline",
                "description": "When a camera goes offline",
                "default": True
            },
            {
                "key": "camera_online",
                "name": "Camera Online",
                "description": "When a camera comes back online",
                "default": False
            },
            {
                "key": "system_alert",
                "name": "System Alert",
                "description": "System-level notifications",
                "default": True
            },
            {
                "key": "daily_summary",
                "name": "Daily Summary",
                "description": "Daily detection summary",
                "default": False
            }
        ]
    }
