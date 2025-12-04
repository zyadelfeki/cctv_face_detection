import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv


class SystemConfig(BaseModel):
    mode: str = Field(default="development")
    debug: bool = Field(default=True)
    gpu_enabled: bool = Field(default=False)
    max_cameras: int = Field(default=4)
    log_level: str = Field(default="INFO")
    temp_dir: str = Field(default="./temp")
    models_dir: str = Field(default="./models")


class DetectionConfig(BaseModel):
    engine: str = Field(default="mtcnn")
    confidence_threshold: float = Field(default=0.9)
    min_face_size: int = Field(default=40)
    scale_factor: float = Field(default=0.709)
    steps_threshold: tuple = Field(default=(0.6, 0.7, 0.7))
    nms_threshold: float = Field(default=0.7)
    margin: int = Field(default=44)
    keep_all: bool = Field(default=True)


class RecognitionConfig(BaseModel):
    engine: str = Field(default="facenet")
    model_name: str = Field(default="vggface2")
    similarity_threshold: float = Field(default=0.4)
    embedding_size: int = Field(default=128)
    device: str = Field(default="cpu")
    batch_size: int = Field(default=16)


class DatabaseConfig(BaseModel):
    type: str = Field(default="postgresql")
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="cctv_surveillance")
    user: str = Field(default="postgres")
    password: str = Field(default="password")
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)
    pool_timeout: int = Field(default=30)
    pool_recycle: int = Field(default=3600)


class RedisConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: Optional[str] = Field(default=None)


class CameraSource(BaseModel):
    name: str
    url: str
    location: Optional[str] = None
    active: bool = True


class CamerasConfig(BaseModel):
    default_fps: int = Field(default=25)
    buffer_size: int = Field(default=1)
    reconnect_interval: int = Field(default=5)
    timeout: int = Field(default=10)
    sources: list[CameraSource] = Field(default_factory=list)


class AlertEmailConfig(BaseModel):
    enabled: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    username: Optional[str] = None
    password: Optional[str] = None
    recipients: list[str] = Field(default_factory=list)


class AlertSMSConfig(BaseModel):
    enabled: bool = False
    twilio_sid: Optional[str] = None
    twilio_token: Optional[str] = None
    from_number: Optional[str] = None
    recipients: list[str] = Field(default_factory=list)


class AlertWebhookConfig(BaseModel):
    enabled: bool = False
    url: Optional[str] = None
    headers: Dict[str, Any] = Field(default_factory=dict)


class AlertingConfig(BaseModel):
    enabled: bool = True
    email: AlertEmailConfig = Field(default_factory=AlertEmailConfig)
    sms: AlertSMSConfig = Field(default_factory=AlertSMSConfig)
    webhook: AlertWebhookConfig = Field(default_factory=AlertWebhookConfig)


class APIAuthConfig(BaseModel):
    enabled: bool = True
    secret_key: str = "change-me"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30


class APIRateLimitConfig(BaseModel):
    enabled: bool = True
    calls: int = 100
    period: int = 60


class APIConfig(BaseModel):
    title: str = "CCTV Face Detection API"
    description: str = "Criminal identification system API"
    version: str = "1.0.0"
    authentication: APIAuthConfig = Field(default_factory=APIAuthConfig)
    rate_limiting: APIRateLimitConfig = Field(default_factory=APIRateLimitConfig)


class PerformanceConfig(BaseModel):
    thread_pool_size: int = 4
    max_queue_size: int = 100
    frame_skip: int = 2
    resize_factor: float = 1.0
    optimization_level: int = 1
    # GPU batching settings
    batch_size: int = 8
    batch_timeout: float = 0.1  # seconds to wait before processing incomplete batch
    num_workers: int = 4
    # Async processing
    use_gpu_batching: bool = True
    prefetch_frames: int = 2


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    rotation: str = "10 MB"
    retention: str = "1 month"
    file: str = "logs/cctv_system.log"
    console: bool = True


class StorageConfig(BaseModel):
    detected_faces_dir: str = "./data/detected_faces"
    criminal_photos_dir: str = "./data/criminal_photos"
    logs_dir: str = "./logs"
    temp_dir: str = "./temp"
    backup_dir: str = "./backups"
    max_storage_gb: int = 100


class SecurityConfig(BaseModel):
    encryption_key: str = "your-encryption-key-32-bytes-long"
    password_salt_rounds: int = 12
    session_timeout_minutes: int = 60
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 15


class WebConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    reload: bool = True
    workers: int = 1
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"])


class RootConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    recognition: RecognitionConfig = Field(default_factory=RecognitionConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    cameras: CamerasConfig = Field(default_factory=CamerasConfig)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    web: WebConfig = Field(default_factory=WebConfig)


class Config:
    def __init__(self, path: Optional[str] = None):
        load_dotenv(override=True)
        self.root_dir = Path(__file__).resolve().parents[2]
        self.config_path = Path(path) if path else self.root_dir / "config" / "config.yaml"
        self.data: RootConfig = self._load()
        self._ensure_dirs()

    def _load(self) -> RootConfig:
        config_dict: Dict[str, Any] = {}
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}
        try:
            return RootConfig(**config_dict)
        except ValidationError as e:
            raise RuntimeError(f"Invalid configuration: {e}")

    def _ensure_dirs(self):
        for p in [
            self.data.storage.detected_faces_dir,
            self.data.storage.criminal_photos_dir,
            self.data.storage.logs_dir,
            self.data.storage.temp_dir,
            self.data.storage.backup_dir,
        ]:
            Path(p).mkdir(parents=True, exist_ok=True)

    def get(self) -> RootConfig:
        return self.data
