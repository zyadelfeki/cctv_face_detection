"""
Enhanced alert service with priority-based routing and multiple channels.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from ..utils.config import Config
from .notifiers import (
    EmailAlerter,
    WebhookAlerter,
    SlackAlerter,
    DiscordAlerter,
    TwilioSMSAlerter,
    PushoverAlerter,
    AlertPriority
)
from .cooldown import CooldownManager


class AlertService:
    """
    Enhanced alert service with:
    - Priority-based routing
    - Multiple notification channels
    - Cooldown management to prevent alert fatigue
    - Async sending for better performance
    """
    
    def __init__(self, config: Config):
        self.config = config
        acfg = config.get().alerting
        self.enabled = acfg.enabled
        
        # Cooldown to prevent duplicate alerts
        self.cooldown = CooldownManager(cooldown_seconds=300)
        
        # Thread pool for async sending
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize notifiers
        self._init_notifiers(acfg)
        
        # Priority thresholds for criminal threat levels
        self.threat_level_priority = {
            "critical": AlertPriority.CRITICAL,
            "high": AlertPriority.HIGH,
            "medium": AlertPriority.MEDIUM,
            "low": AlertPriority.LOW
        }
    
    def _init_notifiers(self, acfg):
        """Initialize all configured notifiers."""
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
            self.webhook = WebhookAlerter(
                url=acfg.webhook.url,
                secret=getattr(acfg.webhook, 'secret', None),
                headers=acfg.webhook.headers
            )
        
        # Slack
        self.slack = None
        if hasattr(acfg, 'slack') and acfg.slack.enabled:
            self.slack = SlackAlerter(
                webhook_url=acfg.slack.webhook_url,
                channel=getattr(acfg.slack, 'channel', None)
            )
        
        # Discord
        self.discord = None
        if hasattr(acfg, 'discord') and acfg.discord.enabled:
            self.discord = DiscordAlerter(
                webhook_url=acfg.discord.webhook_url
            )
        
        # SMS via Twilio
        self.sms = None
        if acfg.sms.enabled:
            self.sms = TwilioSMSAlerter(
                account_sid=acfg.sms.twilio_sid,
                auth_token=acfg.sms.twilio_token,
                from_number=acfg.sms.from_number
            )
        
        # Pushover
        self.pushover = None
        if hasattr(acfg, 'pushover') and acfg.pushover.enabled:
            self.pushover = PushoverAlerter(
                api_token=acfg.pushover.api_token,
                user_key=acfg.pushover.user_key
            )
    
    def _build_email_body(
        self,
        criminal_name: str,
        similarity: float,
        camera_name: str,
        location: Optional[str],
        snapshot_path: Optional[str],
        priority: AlertPriority,
        threat_level: Optional[str] = None
    ) -> str:
        """Build HTML email body."""
        snap_text = f'<img src="cid:snapshot" style="max-width:400px;"/>' if snapshot_path else "No snapshot available"
        location_text = f" at {location}" if location else ""
        
        priority_colors = {
            AlertPriority.LOW: "#28a745",
            AlertPriority.MEDIUM: "#ffc107",
            AlertPriority.HIGH: "#fd7e14",
            AlertPriority.CRITICAL: "#dc3545"
        }
        
        return f"""
        <html>
          <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: {priority_colors[priority]}; color: white; padding: 15px; text-align: center;">
              <h2 style="margin: 0;">🚨 {priority.name} ALERT: Face Match Detected</h2>
            </div>
            <div style="padding: 20px; background: #f8f9fa;">
              <table style="width: 100%; border-collapse: collapse;">
                <tr>
                  <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Criminal:</strong></td>
                  <td style="padding: 10px; border-bottom: 1px solid #ddd;">{criminal_name}</td>
                </tr>
                <tr>
                  <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Threat Level:</strong></td>
                  <td style="padding: 10px; border-bottom: 1px solid #ddd;">{threat_level or 'Unknown'}</td>
                </tr>
                <tr>
                  <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Similarity:</strong></td>
                  <td style="padding: 10px; border-bottom: 1px solid #ddd;">{similarity:.1%}</td>
                </tr>
                <tr>
                  <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Camera:</strong></td>
                  <td style="padding: 10px; border-bottom: 1px solid #ddd;">{camera_name}{location_text}</td>
                </tr>
                <tr>
                  <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Time:</strong></td>
                  <td style="padding: 10px; border-bottom: 1px solid #ddd;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
                </tr>
              </table>
              <div style="margin-top: 20px; text-align: center;">
                {snap_text}
              </div>
            </div>
            <div style="padding: 10px; background: #343a40; color: #fff; text-align: center; font-size: 12px;">
              CCTV Face Detection System | Automated Alert
            </div>
          </body>
        </html>
        """
    
    def _determine_priority(
        self,
        similarity: float,
        threat_level: Optional[str] = None
    ) -> AlertPriority:
        """Determine alert priority based on match and threat level."""
        # Start with threat level if provided
        if threat_level:
            base_priority = self.threat_level_priority.get(
                threat_level.lower(),
                AlertPriority.MEDIUM
            )
        else:
            base_priority = AlertPriority.MEDIUM
        
        # Boost priority for high similarity matches
        if similarity >= 0.95:
            if base_priority.value < AlertPriority.CRITICAL.value:
                return AlertPriority(base_priority.value + 1)
        elif similarity >= 0.85:
            pass  # Keep base priority
        elif similarity >= 0.70:
            if base_priority.value > AlertPriority.LOW.value:
                return AlertPriority(base_priority.value - 1)
        else:
            return AlertPriority.LOW
        
        return base_priority
    
    def send_alert(
        self,
        *,
        criminal_name: str,
        similarity: float,
        camera_name: str,
        location: Optional[str],
        snapshot_path: Optional[str],
        threat_level: Optional[str] = None,
        criminal_id: Optional[int] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """
        Send alert across all configured channels.
        
        Args:
            criminal_name: Name of matched criminal
            similarity: Match similarity score (0-1)
            camera_name: Camera where detection occurred
            location: Physical location of camera
            snapshot_path: Path to detection snapshot image
            threat_level: Criminal's threat level (critical/high/medium/low)
            criminal_id: Database ID of criminal
            additional_data: Any additional context
        """
        if not self.enabled:
            return
        
        # Check cooldown
        cooldown_key = f"{criminal_name}:{camera_name}"
        if not self.cooldown.should_alert(cooldown_key):
            logger.debug(f"Alert for {criminal_name} on {camera_name} suppressed (cooldown)")
            return
        
        # Determine priority
        priority = self._determine_priority(similarity, threat_level)
        
        # Build common payload
        payload = {
            "type": "face_match",
            "criminal": criminal_name,
            "criminal_id": criminal_id,
            "similarity": similarity,
            "camera": camera_name,
            "location": location,
            "snapshot": snapshot_path,
            "threat_level": threat_level,
            "priority": priority.name,
            "timestamp": time.time(),
            "timestamp_iso": datetime.utcnow().isoformat() + "Z",
            **(additional_data or {})
        }
        
        # Send to all channels asynchronously
        self._executor.submit(self._send_all_channels, payload, priority, snapshot_path)
    
    def _send_all_channels(
        self,
        payload: Dict[str, Any],
        priority: AlertPriority,
        snapshot_path: Optional[str]
    ):
        """Send to all configured channels."""
        acfg = self.config.get().alerting
        
        # Email
        if self.email and acfg.email.recipients:
            html_body = self._build_email_body(
                criminal_name=payload["criminal"],
                similarity=payload["similarity"],
                camera_name=payload["camera"],
                location=payload["location"],
                snapshot_path=snapshot_path,
                priority=priority,
                threat_level=payload.get("threat_level")
            )
            attachments = [snapshot_path] if snapshot_path and Path(snapshot_path).exists() else None
            self.email.send(
                acfg.email.recipients,
                subject=f"Face Match: {payload['criminal']} ({payload['similarity']:.0%})",
                html_body=html_body,
                attachments=attachments,
                priority=priority
            )
        
        # Webhook
        if self.webhook:
            self.webhook.send(payload, priority)
        
        # Slack
        if self.slack:
            self.slack.send(payload, priority)
        
        # Discord
        if self.discord:
            self.discord.send(payload, priority)
        
        # SMS (only for HIGH and CRITICAL)
        if self.sms and priority.value >= AlertPriority.HIGH.value:
            sms_recipients = acfg.sms.recipients
            message = (
                f"CCTV Alert: {payload['criminal']} detected on {payload['camera']} "
                f"(similarity: {payload['similarity']:.0%})"
            )
            self.sms.send(sms_recipients, message, priority)
        
        # Pushover (only for HIGH and CRITICAL)
        if self.pushover and priority.value >= AlertPriority.HIGH.value:
            self.pushover.send(
                title=f"Face Match: {payload['criminal']}",
                message=f"Detected on {payload['camera']} with {payload['similarity']:.0%} similarity",
                priority=priority,
                image_path=snapshot_path
            )
        
        logger.info(f"Alert sent for {payload['criminal']} with priority {priority.name}")
    
    def test_all_channels(self) -> Dict[str, bool]:
        """Test all configured notification channels."""
        results = {}
        
        if self.email:
            results["email"] = self.email.test_connection()
        if self.webhook:
            results["webhook"] = self.webhook.test_connection()
        if self.slack:
            results["slack"] = self.slack.test_connection()
        if self.discord:
            results["discord"] = self.discord.test_connection()
        if self.sms:
            results["sms"] = self.sms.test_connection()
        if self.pushover:
            results["pushover"] = self.pushover.test_connection()
        
        return results

