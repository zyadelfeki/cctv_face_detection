"""
Enhanced notification system with multiple channels and priority support.
"""

from __future__ import annotations
import smtplib
import hmac
import hashlib
import json
from abc import ABC, abstractmethod
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
import requests


class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class BaseNotifier(ABC):
    """Base class for all notifiers."""
    
    @abstractmethod
    def send(self, message: Dict[str, Any], priority: AlertPriority = AlertPriority.MEDIUM) -> bool:
        """Send notification."""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test if notifier is properly configured."""
        pass


class EmailAlerter(BaseNotifier):
    """Enhanced email alerter with attachments and priority support."""
    
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        sender: str,
        starttls: bool = True,
        use_ssl: bool = False
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.sender = sender
        self.starttls = starttls
        self.use_ssl = use_ssl

    def send(
        self,
        recipients: List[str],
        subject: str,
        html_body: str,
        attachments: Optional[List[str]] = None,
        priority: AlertPriority = AlertPriority.MEDIUM
    ) -> bool:
        msg = MIMEMultipart('mixed')
        msg['Subject'] = f"[{priority.name}] {subject}"
        msg['From'] = self.sender
        msg['To'] = ", ".join(recipients)
        msg['X-Priority'] = str(5 - priority.value)  # 1 = High, 5 = Low
        
        # HTML body
        html_part = MIMEMultipart('alternative')
        html_part.attach(MIMEText(html_body, 'html'))
        msg.attach(html_part)
        
        # Attachments
        if attachments:
            for filepath in attachments:
                path = Path(filepath)
                if path.exists() and path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                    with open(path, 'rb') as f:
                        img = MIMEImage(f.read())
                        img.add_header('Content-Disposition', 'attachment', filename=path.name)
                        msg.attach(img)

        try:
            if self.use_ssl:
                with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, timeout=10) as server:
                    server.login(self.username, self.password)
                    server.sendmail(self.sender, recipients, msg.as_string())
            else:
                with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                    if self.starttls:
                        server.starttls()
                    server.login(self.username, self.password)
                    server.sendmail(self.sender, recipients, msg.as_string())
            logger.success(f"Alert email sent to {recipients}")
            return True
        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")
            return False
    
    def test_connection(self) -> bool:
        try:
            if self.use_ssl:
                with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, timeout=5) as server:
                    server.login(self.username, self.password)
            else:
                with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=5) as server:
                    if self.starttls:
                        server.starttls()
                    server.login(self.username, self.password)
            return True
        except Exception as e:
            logger.error(f"Email connection test failed: {e}")
            return False


class WebhookAlerter(BaseNotifier):
    """Enhanced webhook alerter with retry and signing."""
    
    def __init__(
        self,
        url: str,
        secret: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        retry_count: int = 3,
        timeout: int = 10
    ):
        self.url = url
        self.secret = secret
        self.headers = headers or {}
        self.retry_count = retry_count
        self.timeout = timeout

    def sign(self, payload: str) -> Dict[str, str]:
        if not self.secret:
            return {}
        signature = hmac.new(self.secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        return {"X-Signature": f"sha256={signature}"}

    def send(self, payload: Dict[str, Any], priority: AlertPriority = AlertPriority.MEDIUM) -> bool:
        payload["priority"] = priority.name
        body = json.dumps(payload)
        headers = {
            "Content-Type": "application/json",
            "X-Priority": str(priority.value),
            **self.headers,
            **self.sign(body)
        }
        
        for attempt in range(self.retry_count):
            try:
                resp = requests.post(self.url, data=body, headers=headers, timeout=self.timeout)
                if 200 <= resp.status_code < 300:
                    logger.success("Alert webhook sent")
                    return True
                else:
                    logger.warning(f"Webhook attempt {attempt + 1} failed: {resp.status_code}")
            except requests.RequestException as e:
                logger.warning(f"Webhook attempt {attempt + 1} error: {e}")
            
            if attempt < self.retry_count - 1:
                asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"Webhook failed after {self.retry_count} attempts")
        return False
    
    def test_connection(self) -> bool:
        try:
            resp = requests.head(self.url, timeout=5)
            return resp.status_code < 500
        except Exception:
            return False


class SlackAlerter(BaseNotifier):
    """Slack notification via webhook."""
    
    def __init__(self, webhook_url: str, channel: Optional[str] = None, username: str = "CCTV Alert Bot"):
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username
    
    def _format_message(self, message: Dict[str, Any], priority: AlertPriority) -> Dict:
        """Format message as Slack blocks."""
        color_map = {
            AlertPriority.LOW: "#36a64f",
            AlertPriority.MEDIUM: "#ffcc00",
            AlertPriority.HIGH: "#ff6600",
            AlertPriority.CRITICAL: "#ff0000"
        }
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🚨 {priority.name} Alert: {message.get('type', 'Unknown')}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Criminal:*\n{message.get('criminal', 'Unknown')}"},
                    {"type": "mrkdwn", "text": f"*Similarity:*\n{message.get('similarity', 0):.2%}"},
                    {"type": "mrkdwn", "text": f"*Camera:*\n{message.get('camera', 'Unknown')}"},
                    {"type": "mrkdwn", "text": f"*Location:*\n{message.get('location', 'N/A')}"}
                ]
            }
        ]
        
        if message.get('snapshot'):
            blocks.append({
                "type": "image",
                "image_url": message['snapshot'],
                "alt_text": "Detection snapshot"
            })
        
        payload = {
            "username": self.username,
            "attachments": [{
                "color": color_map[priority],
                "blocks": blocks
            }]
        }
        
        if self.channel:
            payload["channel"] = self.channel
        
        return payload
    
    def send(self, message: Dict[str, Any], priority: AlertPriority = AlertPriority.MEDIUM) -> bool:
        payload = self._format_message(message, priority)
        
        try:
            resp = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            if resp.status_code == 200:
                logger.success("Slack alert sent")
                return True
            else:
                logger.error(f"Slack webhook failed: {resp.status_code} - {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def test_connection(self) -> bool:
        try:
            resp = requests.post(
                self.webhook_url,
                json={"text": "Test connection from CCTV system"},
                timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False


class DiscordAlerter(BaseNotifier):
    """Discord notification via webhook."""
    
    def __init__(self, webhook_url: str, username: str = "CCTV Alert Bot"):
        self.webhook_url = webhook_url
        self.username = username
    
    def _format_embed(self, message: Dict[str, Any], priority: AlertPriority) -> Dict:
        """Format message as Discord embed."""
        color_map = {
            AlertPriority.LOW: 0x36a64f,
            AlertPriority.MEDIUM: 0xffcc00,
            AlertPriority.HIGH: 0xff6600,
            AlertPriority.CRITICAL: 0xff0000
        }
        
        embed = {
            "title": f"🚨 {priority.name} Alert: Face Match Detected",
            "color": color_map[priority],
            "fields": [
                {"name": "Criminal", "value": message.get('criminal', 'Unknown'), "inline": True},
                {"name": "Similarity", "value": f"{message.get('similarity', 0):.2%}", "inline": True},
                {"name": "Camera", "value": message.get('camera', 'Unknown'), "inline": True},
                {"name": "Location", "value": message.get('location', 'N/A'), "inline": True}
            ],
            "timestamp": message.get('timestamp_iso')
        }
        
        if message.get('snapshot_url'):
            embed["image"] = {"url": message['snapshot_url']}
        
        return {
            "username": self.username,
            "embeds": [embed]
        }
    
    def send(self, message: Dict[str, Any], priority: AlertPriority = AlertPriority.MEDIUM) -> bool:
        payload = self._format_embed(message, priority)
        
        try:
            resp = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            if 200 <= resp.status_code < 300:
                logger.success("Discord alert sent")
                return True
            else:
                logger.error(f"Discord webhook failed: {resp.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False
    
    def test_connection(self) -> bool:
        try:
            # Discord webhooks support GET to check validity
            resp = requests.get(self.webhook_url, timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


class TwilioSMSAlerter(BaseNotifier):
    """SMS alerts via Twilio."""
    
    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.base_url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    
    def send(
        self,
        recipients: List[str],
        message: str,
        priority: AlertPriority = AlertPriority.MEDIUM
    ) -> bool:
        success = True
        prefix = f"[{priority.name}] " if priority == AlertPriority.CRITICAL else ""
        
        for phone in recipients:
            try:
                resp = requests.post(
                    self.base_url,
                    auth=(self.account_sid, self.auth_token),
                    data={
                        "From": self.from_number,
                        "To": phone,
                        "Body": f"{prefix}{message}"
                    },
                    timeout=10
                )
                if resp.status_code not in [200, 201]:
                    logger.error(f"SMS to {phone} failed: {resp.text}")
                    success = False
                else:
                    logger.success(f"SMS sent to {phone}")
            except Exception as e:
                logger.error(f"Failed to send SMS to {phone}: {e}")
                success = False
        
        return success
    
    def test_connection(self) -> bool:
        try:
            resp = requests.get(
                f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}.json",
                auth=(self.account_sid, self.auth_token),
                timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False


class PushoverAlerter(BaseNotifier):
    """Push notifications via Pushover."""
    
    API_URL = "https://api.pushover.net/1/messages.json"
    
    def __init__(self, api_token: str, user_key: str, device: Optional[str] = None):
        self.api_token = api_token
        self.user_key = user_key
        self.device = device
    
    def send(
        self,
        title: str,
        message: str,
        priority: AlertPriority = AlertPriority.MEDIUM,
        url: Optional[str] = None,
        image_path: Optional[str] = None
    ) -> bool:
        # Map to Pushover priority (-2 to 2)
        priority_map = {
            AlertPriority.LOW: -1,
            AlertPriority.MEDIUM: 0,
            AlertPriority.HIGH: 1,
            AlertPriority.CRITICAL: 2
        }
        
        data = {
            "token": self.api_token,
            "user": self.user_key,
            "title": title,
            "message": message,
            "priority": priority_map[priority]
        }
        
        if priority == AlertPriority.CRITICAL:
            data["retry"] = 60
            data["expire"] = 3600
        
        if self.device:
            data["device"] = self.device
        
        if url:
            data["url"] = url
        
        files = None
        if image_path and Path(image_path).exists():
            files = {"attachment": open(image_path, "rb")}
        
        try:
            resp = requests.post(self.API_URL, data=data, files=files, timeout=10)
            if files:
                files["attachment"].close()
            
            if resp.status_code == 200:
                logger.success("Pushover notification sent")
                return True
            else:
                logger.error(f"Pushover failed: {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Failed to send Pushover notification: {e}")
            return False
    
    def test_connection(self) -> bool:
        try:
            resp = requests.post(
                "https://api.pushover.net/1/users/validate.json",
                data={"token": self.api_token, "user": self.user_key},
                timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False

