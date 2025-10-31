from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import time

from loguru import logger

from ..utils.config import Config
from .notifiers import EmailAlerter, WebhookAlerter
from .cooldown import CooldownManager


class AlertService:
    def __init__(self, config: Config):
        self.config = config
        acfg = config.get().alerting
        self.enabled = acfg.enabled
        self.cooldown = CooldownManager(cooldown_seconds=300)
        # Email
        self.email = None
        if acfg.email.enabled:
            self.email = EmailAlerter(
                smtp_server=acfg.email.smtp_server,
                smtp_port=acfg.email.smtp_port,
                username=acfg.email.username,
                password=acfg.email.password,
                sender=acfg.email.username,
                starttls=True,
            )
        # Webhook
        self.webhook = None
        if acfg.webhook.enabled:
            self.webhook = WebhookAlerter(url=acfg.webhook.url, headers=acfg.webhook.headers)

    def _build_email_body(self, criminal_name: str, similarity: float, camera_name: str, location: Optional[str], snapshot_path: Optional[str]) -> str:
        snap_text = f"Snapshot saved: {snapshot_path}" if snapshot_path else "No snapshot available"
        location_text = f" at {location}" if location else ""
        return f"""
        <html>
          <body>
            <h3>Face Match Alert</h3>
            <p>Criminal <b>{criminal_name}</b> detected on camera <b>{camera_name}</b>{location_text}.</p>
            <p>Similarity score: {similarity:.3f}</p>
            <p>{snap_text}</p>
            <p>Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
          </body>
        </html>
        """

    def send_alert(self, *, criminal_name: str, similarity: float, camera_name: str, location: Optional[str], snapshot_path: Optional[str]):
        if not self.enabled:
            return
        body = self._build_email_body(criminal_name, similarity, camera_name, location, snapshot_path)
        acfg = self.config.get().alerting
        if self.email and acfg.email.recipients:
            self.email.send(acfg.email.recipients, subject=f"Face Match: {criminal_name} ({similarity:.2f})", html_body=body)
        if self.webhook:
            payload = {
                "type": "face_match",
                "criminal": criminal_name,
                "similarity": similarity,
                "camera": camera_name,
                "location": location,
                "snapshot": snapshot_path,
                "timestamp": time.time(),
            }
            self.webhook.send(payload)
