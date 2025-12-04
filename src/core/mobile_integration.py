"""
Mobile App Integration Module

Push notifications via Firebase Cloud Messaging (FCM),
mobile-optimized API endpoints, and live streaming support.

Features:
- Firebase Cloud Messaging for push notifications
- Mobile device registration and management
- Real-time alert streaming
- Bandwidth-adaptive video streaming
- Offline sync support

Author: CCTV Face Detection System
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import hashlib
import hmac
import time
from pathlib import Path

import aiohttp
import numpy as np

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Types of push notifications"""
    FACE_DETECTED = "face_detected"
    UNKNOWN_PERSON = "unknown_person"
    LIVENESS_FAIL = "liveness_fail"
    SECURITY_ALERT = "security_alert"
    PERSON_OF_INTEREST = "person_of_interest"
    CAMERA_OFFLINE = "camera_offline"
    CAMERA_ONLINE = "camera_online"
    SYSTEM_ALERT = "system_alert"
    DAILY_SUMMARY = "daily_summary"


class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MobileDevice:
    """Registered mobile device"""
    device_id: str
    fcm_token: str
    user_id: str
    device_name: str
    platform: str  # 'ios', 'android'
    app_version: str
    registered_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    notification_settings: Dict[str, bool] = field(default_factory=dict)
    timezone: str = "UTC"
    is_active: bool = True
    
    def __post_init__(self):
        if not self.notification_settings:
            # Default: all notifications enabled
            self.notification_settings = {
                nt.value: True for nt in NotificationType
            }


@dataclass
class Notification:
    """Push notification data"""
    notification_id: str
    notification_type: NotificationType
    title: str
    body: str
    data: Dict[str, Any] = field(default_factory=dict)
    image_url: Optional[str] = None
    priority: NotificationPriority = NotificationPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    sent_to: List[str] = field(default_factory=list)
    click_action: Optional[str] = None
    
    def to_fcm_payload(self, device: MobileDevice) -> Dict[str, Any]:
        """Convert to FCM message format"""
        payload = {
            "message": {
                "token": device.fcm_token,
                "notification": {
                    "title": self.title,
                    "body": self.body,
                },
                "data": {
                    "notification_id": self.notification_id,
                    "type": self.notification_type.value,
                    "timestamp": self.created_at.isoformat(),
                    **{k: str(v) for k, v in self.data.items()}
                },
                "android": {
                    "priority": "high" if self.priority in [NotificationPriority.HIGH, NotificationPriority.CRITICAL] else "normal",
                    "notification": {
                        "sound": "default",
                        "click_action": self.click_action or "FLUTTER_NOTIFICATION_CLICK"
                    }
                },
                "apns": {
                    "payload": {
                        "aps": {
                            "sound": "default",
                            "badge": 1
                        }
                    }
                }
            }
        }
        
        if self.image_url:
            payload["message"]["notification"]["image"] = self.image_url
            
        return payload


class FirebaseCloudMessaging:
    """
    Firebase Cloud Messaging client for push notifications.
    
    Requires service account credentials from Firebase Console.
    """
    
    FCM_URL = "https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    
    def __init__(
        self,
        project_id: str,
        service_account_path: Optional[str] = None,
        service_account_json: Optional[Dict] = None
    ):
        self.project_id = project_id
        self.service_account = self._load_service_account(
            service_account_path, service_account_json
        )
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0
        
    def _load_service_account(
        self,
        path: Optional[str],
        json_data: Optional[Dict]
    ) -> Dict:
        """Load service account credentials"""
        if json_data:
            return json_data
        if path:
            with open(path, 'r') as f:
                return json.load(f)
        # Fallback: check environment
        import os
        env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if env_path and Path(env_path).exists():
            with open(env_path, 'r') as f:
                return json.load(f)
        raise ValueError("Firebase service account credentials required")
    
    async def _get_access_token(self) -> str:
        """Get OAuth2 access token for FCM API"""
        import jwt
        
        if self._access_token and time.time() < self._token_expiry:
            return self._access_token
            
        now = int(time.time())
        payload = {
            "iss": self.service_account["client_email"],
            "scope": "https://www.googleapis.com/auth/firebase.messaging",
            "aud": self.TOKEN_URL,
            "iat": now,
            "exp": now + 3600
        }
        
        token = jwt.encode(
            payload,
            self.service_account["private_key"],
            algorithm="RS256"
        )
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.TOKEN_URL,
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "assertion": token
                }
            ) as response:
                data = await response.json()
                self._access_token = data["access_token"]
                self._token_expiry = now + data.get("expires_in", 3500)
                return self._access_token
    
    async def send_notification(
        self,
        notification: Notification,
        device: MobileDevice
    ) -> bool:
        """
        Send push notification to a device.
        
        Args:
            notification: Notification to send
            device: Target device
            
        Returns:
            True if successful
        """
        # Check if device wants this notification type
        if not device.notification_settings.get(notification.notification_type.value, True):
            logger.debug(f"Device {device.device_id} has disabled {notification.notification_type.value}")
            return False
            
        try:
            token = await self._get_access_token()
            payload = notification.to_fcm_payload(device)
            
            async with aiohttp.ClientSession() as session:
                url = self.FCM_URL.format(project_id=self.project_id)
                async with session.post(
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json"
                    }
                ) as response:
                    if response.status == 200:
                        notification.sent_to.append(device.device_id)
                        logger.info(f"Sent notification to {device.device_id}")
                        return True
                    else:
                        error = await response.text()
                        logger.error(f"FCM error: {error}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False
    
    async def send_to_topic(
        self,
        notification: Notification,
        topic: str
    ) -> bool:
        """Send notification to all devices subscribed to a topic"""
        try:
            token = await self._get_access_token()
            
            payload = {
                "message": {
                    "topic": topic,
                    "notification": {
                        "title": notification.title,
                        "body": notification.body
                    },
                    "data": {
                        "notification_id": notification.notification_id,
                        "type": notification.notification_type.value,
                        **{k: str(v) for k, v in notification.data.items()}
                    }
                }
            }
            
            async with aiohttp.ClientSession() as session:
                url = self.FCM_URL.format(project_id=self.project_id)
                async with session.post(
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json"
                    }
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"Failed to send topic notification: {e}")
            return False


class DeviceManager:
    """
    Manages registered mobile devices and their notification preferences.
    """
    
    def __init__(self, db_session=None):
        self.devices: Dict[str, MobileDevice] = {}
        self.user_devices: Dict[str, Set[str]] = {}  # user_id -> device_ids
        self.db_session = db_session
        
    def register_device(
        self,
        device_id: str,
        fcm_token: str,
        user_id: str,
        device_name: str,
        platform: str,
        app_version: str,
        timezone: str = "UTC"
    ) -> MobileDevice:
        """
        Register a new mobile device.
        
        Args:
            device_id: Unique device identifier
            fcm_token: Firebase Cloud Messaging token
            user_id: Associated user ID
            device_name: Human-readable device name
            platform: 'ios' or 'android'
            app_version: App version string
            timezone: Device timezone
            
        Returns:
            Registered MobileDevice
        """
        device = MobileDevice(
            device_id=device_id,
            fcm_token=fcm_token,
            user_id=user_id,
            device_name=device_name,
            platform=platform,
            app_version=app_version,
            timezone=timezone
        )
        
        self.devices[device_id] = device
        
        if user_id not in self.user_devices:
            self.user_devices[user_id] = set()
        self.user_devices[user_id].add(device_id)
        
        logger.info(f"Registered device {device_id} for user {user_id}")
        return device
    
    def update_fcm_token(self, device_id: str, new_token: str) -> bool:
        """Update FCM token for a device (tokens can expire/refresh)"""
        if device_id in self.devices:
            self.devices[device_id].fcm_token = new_token
            self.devices[device_id].last_seen = datetime.now()
            return True
        return False
    
    def update_notification_settings(
        self,
        device_id: str,
        settings: Dict[str, bool]
    ) -> bool:
        """Update notification preferences for a device"""
        if device_id in self.devices:
            self.devices[device_id].notification_settings.update(settings)
            return True
        return False
    
    def deactivate_device(self, device_id: str) -> bool:
        """Deactivate a device (user logged out or uninstalled)"""
        if device_id in self.devices:
            self.devices[device_id].is_active = False
            return True
        return False
    
    def get_user_devices(self, user_id: str) -> List[MobileDevice]:
        """Get all active devices for a user"""
        device_ids = self.user_devices.get(user_id, set())
        return [
            self.devices[did] 
            for did in device_ids 
            if did in self.devices and self.devices[did].is_active
        ]
    
    def get_all_active_devices(self) -> List[MobileDevice]:
        """Get all active devices"""
        return [d for d in self.devices.values() if d.is_active]


class AlertNotificationService:
    """
    High-level service for sending face detection alerts.
    
    Integrates with the face detection system to send
    real-time notifications for security events.
    """
    
    def __init__(
        self,
        fcm_client: FirebaseCloudMessaging,
        device_manager: DeviceManager,
        image_base_url: str = "http://localhost:8000"
    ):
        self.fcm = fcm_client
        self.devices = device_manager
        self.image_base_url = image_base_url
        self.notification_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        
        # Rate limiting: max notifications per minute per device
        self._rate_limits: Dict[str, List[float]] = {}
        self.rate_limit_per_minute = 10
        
    def _generate_notification_id(self) -> str:
        """Generate unique notification ID"""
        return hashlib.sha256(
            f"{time.time()}-{id(self)}".encode()
        ).hexdigest()[:16]
    
    def _check_rate_limit(self, device_id: str) -> bool:
        """Check if device is within rate limit"""
        now = time.time()
        if device_id not in self._rate_limits:
            self._rate_limits[device_id] = []
        
        # Remove old timestamps (older than 1 minute)
        self._rate_limits[device_id] = [
            ts for ts in self._rate_limits[device_id]
            if now - ts < 60
        ]
        
        if len(self._rate_limits[device_id]) >= self.rate_limit_per_minute:
            return False
            
        self._rate_limits[device_id].append(now)
        return True
    
    async def start(self):
        """Start the notification worker"""
        self._running = True
        self._worker_task = asyncio.create_task(self._notification_worker())
        logger.info("Alert notification service started")
    
    async def stop(self):
        """Stop the notification worker"""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Alert notification service stopped")
    
    async def _notification_worker(self):
        """Background worker to process notification queue"""
        while self._running:
            try:
                notification, user_ids = await asyncio.wait_for(
                    self.notification_queue.get(),
                    timeout=1.0
                )
                
                # Send to all devices of specified users
                for user_id in user_ids:
                    devices = self.devices.get_user_devices(user_id)
                    for device in devices:
                        if self._check_rate_limit(device.device_id):
                            await self.fcm.send_notification(notification, device)
                            
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Notification worker error: {e}")
    
    async def notify_face_detected(
        self,
        person_name: str,
        camera_name: str,
        confidence: float,
        snapshot_path: Optional[str] = None,
        user_ids: Optional[List[str]] = None
    ):
        """
        Send notification for known face detection.
        
        Args:
            person_name: Name of detected person
            camera_name: Camera that detected the face
            confidence: Detection confidence
            snapshot_path: Path to snapshot image
            user_ids: Users to notify (None = all)
        """
        notification = Notification(
            notification_id=self._generate_notification_id(),
            notification_type=NotificationType.FACE_DETECTED,
            title=f"Face Detected: {person_name}",
            body=f"Detected at {camera_name} with {confidence:.1%} confidence",
            data={
                "person_name": person_name,
                "camera_name": camera_name,
                "confidence": confidence
            },
            image_url=f"{self.image_base_url}/snapshots/{snapshot_path}" if snapshot_path else None,
            priority=NotificationPriority.NORMAL,
            click_action="VIEW_DETECTION"
        )
        
        target_users = user_ids or [d.user_id for d in self.devices.get_all_active_devices()]
        await self.notification_queue.put((notification, target_users))
    
    async def notify_unknown_person(
        self,
        camera_name: str,
        track_id: str,
        snapshot_path: Optional[str] = None,
        user_ids: Optional[List[str]] = None
    ):
        """Send notification for unknown person detection"""
        notification = Notification(
            notification_id=self._generate_notification_id(),
            notification_type=NotificationType.UNKNOWN_PERSON,
            title="Unknown Person Detected",
            body=f"Unrecognized face at {camera_name}",
            data={
                "camera_name": camera_name,
                "track_id": track_id
            },
            image_url=f"{self.image_base_url}/snapshots/{snapshot_path}" if snapshot_path else None,
            priority=NotificationPriority.HIGH,
            click_action="VIEW_UNKNOWN"
        )
        
        target_users = user_ids or [d.user_id for d in self.devices.get_all_active_devices()]
        await self.notification_queue.put((notification, target_users))
    
    async def notify_liveness_failure(
        self,
        camera_name: str,
        failure_reason: str,
        snapshot_path: Optional[str] = None,
        user_ids: Optional[List[str]] = None
    ):
        """Send notification for liveness check failure (potential spoof)"""
        notification = Notification(
            notification_id=self._generate_notification_id(),
            notification_type=NotificationType.LIVENESS_FAIL,
            title="⚠️ Potential Spoof Attempt",
            body=f"Liveness check failed at {camera_name}: {failure_reason}",
            data={
                "camera_name": camera_name,
                "failure_reason": failure_reason
            },
            image_url=f"{self.image_base_url}/snapshots/{snapshot_path}" if snapshot_path else None,
            priority=NotificationPriority.CRITICAL,
            click_action="VIEW_SECURITY_ALERT"
        )
        
        target_users = user_ids or [d.user_id for d in self.devices.get_all_active_devices()]
        await self.notification_queue.put((notification, target_users))
    
    async def notify_person_of_interest(
        self,
        person_name: str,
        camera_name: str,
        alert_level: str,
        snapshot_path: Optional[str] = None,
        user_ids: Optional[List[str]] = None
    ):
        """Send high-priority alert for person of interest"""
        notification = Notification(
            notification_id=self._generate_notification_id(),
            notification_type=NotificationType.PERSON_OF_INTEREST,
            title=f"🚨 ALERT: {person_name}",
            body=f"Person of interest detected at {camera_name}. Alert level: {alert_level}",
            data={
                "person_name": person_name,
                "camera_name": camera_name,
                "alert_level": alert_level
            },
            image_url=f"{self.image_base_url}/snapshots/{snapshot_path}" if snapshot_path else None,
            priority=NotificationPriority.CRITICAL,
            click_action="VIEW_POI_ALERT"
        )
        
        target_users = user_ids or [d.user_id for d in self.devices.get_all_active_devices()]
        await self.notification_queue.put((notification, target_users))
    
    async def notify_camera_status(
        self,
        camera_name: str,
        is_online: bool,
        user_ids: Optional[List[str]] = None
    ):
        """Send notification for camera status change"""
        notification = Notification(
            notification_id=self._generate_notification_id(),
            notification_type=NotificationType.CAMERA_ONLINE if is_online else NotificationType.CAMERA_OFFLINE,
            title=f"Camera {'Online' if is_online else 'Offline'}",
            body=f"{camera_name} is now {'online' if is_online else 'offline'}",
            data={
                "camera_name": camera_name,
                "status": "online" if is_online else "offline"
            },
            priority=NotificationPriority.HIGH if not is_online else NotificationPriority.NORMAL
        )
        
        target_users = user_ids or [d.user_id for d in self.devices.get_all_active_devices()]
        await self.notification_queue.put((notification, target_users))


class MobileStreamAdapter:
    """
    Adapts video streams for mobile viewing with bandwidth optimization.
    
    Features:
    - Adaptive bitrate streaming
    - Frame rate adjustment based on network conditions
    - JPEG quality optimization
    - HLS/DASH support for native mobile players
    """
    
    def __init__(
        self,
        frame_callback: Callable[[], np.ndarray],
        target_fps: int = 15,
        jpeg_quality: int = 70
    ):
        self.frame_callback = frame_callback
        self.target_fps = target_fps
        self.jpeg_quality = jpeg_quality
        self.clients: Dict[str, Dict] = {}  # client_id -> settings
        
    def register_client(
        self,
        client_id: str,
        network_type: str = "wifi",  # 'wifi', '4g', '3g', 'slow'
        screen_width: int = 720
    ) -> Dict:
        """
        Register a mobile client with its network conditions.
        
        Returns optimal streaming settings.
        """
        # Adaptive settings based on network
        settings = {
            "wifi": {"fps": 15, "quality": 75, "width": min(1280, screen_width)},
            "4g": {"fps": 10, "quality": 60, "width": min(720, screen_width)},
            "3g": {"fps": 5, "quality": 40, "width": min(480, screen_width)},
            "slow": {"fps": 2, "quality": 30, "width": min(320, screen_width)}
        }
        
        client_settings = settings.get(network_type, settings["4g"])
        self.clients[client_id] = client_settings
        
        return client_settings
    
    def update_network_conditions(
        self,
        client_id: str,
        latency_ms: float,
        bandwidth_kbps: float
    ) -> Dict:
        """
        Dynamically adjust settings based on measured network conditions.
        
        Args:
            client_id: Client identifier
            latency_ms: Round-trip latency
            bandwidth_kbps: Estimated bandwidth
            
        Returns:
            Updated settings
        """
        if client_id not in self.clients:
            return self.register_client(client_id)
            
        settings = self.clients[client_id]
        
        # Adjust based on latency
        if latency_ms > 500:
            settings["fps"] = min(settings["fps"], 5)
            settings["quality"] = min(settings["quality"], 40)
        elif latency_ms > 200:
            settings["fps"] = min(settings["fps"], 10)
            settings["quality"] = min(settings["quality"], 60)
            
        # Adjust based on bandwidth
        if bandwidth_kbps < 500:
            settings["quality"] = 30
            settings["width"] = 320
        elif bandwidth_kbps < 1000:
            settings["quality"] = 50
            settings["width"] = 480
            
        return settings
    
    async def generate_stream(self, client_id: str):
        """
        Generate MJPEG stream for mobile client.
        
        Yields:
            JPEG frames optimized for client
        """
        import cv2
        
        settings = self.clients.get(client_id, {"fps": 10, "quality": 60, "width": 720})
        frame_delay = 1.0 / settings["fps"]
        
        while True:
            try:
                frame = self.frame_callback()
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Resize for mobile
                h, w = frame.shape[:2]
                target_width = settings["width"]
                if w > target_width:
                    scale = target_width / w
                    frame = cv2.resize(
                        frame, 
                        (target_width, int(h * scale)),
                        interpolation=cv2.INTER_LINEAR
                    )
                
                # Encode with quality setting
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, settings["quality"]]
                _, jpeg = cv2.imencode('.jpg', frame, encode_params)
                
                yield jpeg.tobytes()
                
                await asyncio.sleep(frame_delay)
                
            except Exception as e:
                logger.error(f"Stream error for {client_id}: {e}")
                await asyncio.sleep(1.0)


class OfflineSyncManager:
    """
    Manages offline data synchronization for mobile app.
    
    When devices are offline, stores pending updates locally
    and syncs when connection is restored.
    """
    
    def __init__(self):
        self.pending_events: Dict[str, List[Dict]] = {}  # device_id -> events
        self.last_sync: Dict[str, datetime] = {}
        
    def queue_event(self, device_id: str, event: Dict):
        """Queue an event for later sync"""
        if device_id not in self.pending_events:
            self.pending_events[device_id] = []
        
        event["queued_at"] = datetime.now().isoformat()
        self.pending_events[device_id].append(event)
    
    def get_pending_events(self, device_id: str, since: Optional[datetime] = None) -> List[Dict]:
        """Get pending events for a device since last sync"""
        events = self.pending_events.get(device_id, [])
        
        if since:
            events = [
                e for e in events
                if datetime.fromisoformat(e["queued_at"]) > since
            ]
            
        return events
    
    def mark_synced(self, device_id: str, event_ids: List[str]):
        """Mark events as synced"""
        if device_id in self.pending_events:
            self.pending_events[device_id] = [
                e for e in self.pending_events[device_id]
                if e.get("event_id") not in event_ids
            ]
        self.last_sync[device_id] = datetime.now()
    
    def get_sync_payload(
        self,
        device_id: str,
        max_events: int = 100
    ) -> Dict[str, Any]:
        """
        Generate sync payload for device.
        
        Returns:
            Dict with events, detection summaries, settings updates
        """
        events = self.get_pending_events(device_id)[:max_events]
        
        return {
            "sync_timestamp": datetime.now().isoformat(),
            "events": events,
            "has_more": len(self.pending_events.get(device_id, [])) > max_events
        }


# Factory function for easy setup
def create_mobile_notification_system(
    firebase_project_id: str,
    firebase_credentials_path: Optional[str] = None,
    firebase_credentials_json: Optional[Dict] = None,
    image_base_url: str = "http://localhost:8000"
) -> tuple:
    """
    Create and configure mobile notification system.
    
    Args:
        firebase_project_id: Firebase project ID
        firebase_credentials_path: Path to service account JSON
        firebase_credentials_json: Service account dict (alternative to path)
        image_base_url: Base URL for snapshot images
        
    Returns:
        (fcm_client, device_manager, alert_service)
    """
    fcm = FirebaseCloudMessaging(
        project_id=firebase_project_id,
        service_account_path=firebase_credentials_path,
        service_account_json=firebase_credentials_json
    )
    
    device_manager = DeviceManager()
    
    alert_service = AlertNotificationService(
        fcm_client=fcm,
        device_manager=device_manager,
        image_base_url=image_base_url
    )
    
    return fcm, device_manager, alert_service
