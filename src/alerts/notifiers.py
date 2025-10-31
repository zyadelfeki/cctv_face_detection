from __future__ import annotations
import smtplib
import hmac
import hashlib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Optional

from loguru import logger


class EmailAlerter:
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str, sender: str, starttls: bool = True):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.sender = sender
        self.starttls = starttls

    def send(self, recipients: list[str], subject: str, html_body: str):
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = self.sender
        msg['To'] = ", ".join(recipients)
        part = MIMEText(html_body, 'html')
        msg.attach(part)

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                if self.starttls:
                    server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.sender, recipients, msg.as_string())
            logger.success(f"Alert email sent to {recipients}")
        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")


class WebhookAlerter:
    def __init__(self, url: str, secret: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
        self.url = url
        self.secret = secret
        self.headers = headers or {}

    def sign(self, payload: str) -> Dict[str, str]:
        if not self.secret:
            return {}
        signature = hmac.new(self.secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        return {"X-Signature": signature}

    def send(self, payload: Dict[str, any]):
        import json, requests
        body = json.dumps(payload)
        headers = {"Content-Type": "application/json", **self.headers, **self.sign(body)}
        try:
            resp = requests.post(self.url, data=body, headers=headers, timeout=5)
            if resp.status_code >= 200 and resp.status_code < 300:
                logger.success("Alert webhook sent")
            else:
                logger.error(f"Webhook failed with status {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
