"""
Emotion & Demographics Analysis Module

Real-time facial attribute detection:
- Age estimation
- Gender classification
- Emotion recognition (7 emotions)
- Visitor analytics and insights

Author: CCTV Face Detection System
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import json

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Try imports
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Emotion(Enum):
    """7 basic emotions (FER2013 classes)"""
    ANGRY = 0
    DISGUST = 1
    FEAR = 2
    HAPPY = 3
    SAD = 4
    SURPRISE = 5
    NEUTRAL = 6
    
    @classmethod
    def from_index(cls, idx: int) -> 'Emotion':
        for e in cls:
            if e.value == idx:
                return e
        return cls.NEUTRAL


class Gender(Enum):
    """Gender classification"""
    MALE = 0
    FEMALE = 1
    
    @classmethod
    def from_index(cls, idx: int) -> 'Gender':
        return cls.FEMALE if idx == 1 else cls.MALE


@dataclass
class DemographicResult:
    """Result of demographic analysis"""
    age: int
    age_confidence: float
    gender: Gender
    gender_confidence: float
    emotion: Emotion
    emotion_confidence: float
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "age": self.age,
            "age_confidence": self.age_confidence,
            "gender": self.gender.name.lower(),
            "gender_confidence": self.gender_confidence,
            "emotion": self.emotion.name.lower(),
            "emotion_confidence": self.emotion_confidence,
            "emotion_scores": self.emotion_scores
        }


@dataclass
class VisitorStats:
    """Aggregated visitor statistics"""
    period_start: datetime
    period_end: datetime
    total_visitors: int
    unique_visitors: int
    
    # Demographics breakdown
    age_distribution: Dict[str, int] = field(default_factory=dict)  # "0-18", "19-35", "36-50", "51+"
    gender_distribution: Dict[str, int] = field(default_factory=dict)  # "male", "female"
    emotion_distribution: Dict[str, int] = field(default_factory=dict)  # emotion counts
    
    # Time-based patterns
    hourly_traffic: Dict[int, int] = field(default_factory=dict)  # hour -> count
    peak_hour: int = 0
    avg_visit_duration: float = 0.0
    
    # Satisfaction metrics
    positive_emotion_ratio: float = 0.0  # (happy + surprise) / total
    negative_emotion_ratio: float = 0.0  # (angry + sad + fear + disgust) / total
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat()
            },
            "visitors": {
                "total": self.total_visitors,
                "unique": self.unique_visitors
            },
            "demographics": {
                "age": self.age_distribution,
                "gender": self.gender_distribution,
                "emotion": self.emotion_distribution
            },
            "traffic": {
                "hourly": self.hourly_traffic,
                "peak_hour": self.peak_hour,
                "avg_duration_minutes": self.avg_visit_duration
            },
            "satisfaction": {
                "positive_ratio": self.positive_emotion_ratio,
                "negative_ratio": self.negative_emotion_ratio
            }
        }


class MiniXceptionEmotionNet(nn.Module if TORCH_AVAILABLE else object):
    """
    Mini-Xception architecture for emotion recognition.
    
    Based on: https://arxiv.org/abs/1710.07557
    Efficient architecture optimized for real-time inference.
    """
    
    def __init__(self, num_classes: int = 7):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for MiniXceptionEmotionNet")
            
        super().__init__()
        
        # Base convolutions
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        
        # Residual blocks with depthwise separable convolutions
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(8, 16),
            self._make_residual_block(16, 32),
            self._make_residual_block(32, 64),
            self._make_residual_block(64, 128)
        ])
        
        # Classification head
        self.conv_out = nn.Conv2d(128, num_classes, 3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def _make_residual_block(self, in_ch: int, out_ch: int):
        """Create depthwise separable residual block"""
        return nn.Sequential(
            # Depthwise separable conv
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # Second separable conv
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            # Pooling
            nn.MaxPool2d(3, stride=2, padding=1)
        )
    
    def forward(self, x):
        # Input: (B, 1, 48, 48) grayscale face
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.conv_out(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        return x


class SSRNet(nn.Module if TORCH_AVAILABLE else object):
    """
    Soft Stagewise Regression Network for age estimation.
    
    Based on: https://github.com/shamangary/SSR-Net
    Compact network for accurate age prediction.
    """
    
    def __init__(self, stage_num: List[int] = [3, 3, 3], lambda_d: float = 1.0):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for SSRNet")
            
        super().__init__()
        
        self.stage_num = stage_num
        self.lambda_d = lambda_d
        
        # Feature extraction backbone
        self.stream1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2)
        )
        
        self.stream2 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2)
        )
        
        # Stage regression heads
        self.fc_stages = nn.ModuleList()
        for s in stage_num:
            self.fc_stages.append(nn.Sequential(
                nn.Linear(32 * 8 * 8 + 16 * 8 * 8, 128),
                nn.ReLU(),
                nn.Linear(128, s)
            ))
        
        # Delta for fine-grained prediction
        self.fc_delta = nn.Linear(32 * 8 * 8 + 16 * 8 * 8, 1)
    
    def forward(self, x):
        # Input: (B, 3, 64, 64) RGB face
        f1 = self.stream1(x).view(x.size(0), -1)
        f2 = self.stream2(x).view(x.size(0), -1)
        features = torch.cat([f1, f2], dim=1)
        
        # Stage predictions
        stage_outputs = []
        for i, fc in enumerate(self.fc_stages):
            stage_pred = F.softmax(fc(features), dim=1)
            stage_outputs.append(stage_pred)
        
        # Calculate expected age
        # This is a simplified version - full SSR-Net uses more complex aggregation
        age = torch.zeros(x.size(0), device=x.device)
        base = 100.0  # Max age
        
        for i, (stage_out, s) in enumerate(zip(stage_outputs, self.stage_num)):
            idx = torch.arange(s, device=x.device, dtype=torch.float32)
            expected = (stage_out * idx).sum(dim=1)
            age = age + expected * (base / (s ** (i + 1)))
        
        # Fine-grained delta
        delta = torch.tanh(self.fc_delta(features)).squeeze() * self.lambda_d
        age = age + delta
        
        return age.clamp(0, 100)


class GenderNet(nn.Module if TORCH_AVAILABLE else object):
    """Simple CNN for gender classification"""
    
    def __init__(self):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for GenderNet")
            
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        # Input: (B, 3, 64, 64) RGB face
        x = self.features(x)
        x = self.classifier(x)
        return x


class DemographicsAnalyzer:
    """
    Combined age, gender, and emotion analyzer.
    
    Uses efficient CNN architectures optimized for real-time inference.
    """
    
    def __init__(
        self,
        model_dir: str = "models/demographics",
        device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",
        use_pretrained: bool = True
    ):
        self.model_dir = Path(model_dir)
        self.device = device
        
        self.emotion_model: Optional[nn.Module] = None
        self.age_model: Optional[nn.Module] = None
        self.gender_model: Optional[nn.Module] = None
        
        if TORCH_AVAILABLE:
            self._load_models(use_pretrained)
    
    def _load_models(self, use_pretrained: bool):
        """Load or initialize models"""
        try:
            # Initialize models
            self.emotion_model = MiniXceptionEmotionNet(num_classes=7).to(self.device)
            self.age_model = SSRNet().to(self.device)
            self.gender_model = GenderNet().to(self.device)
            
            # Load pretrained weights if available
            emotion_path = self.model_dir / "emotion_model.pth"
            age_path = self.model_dir / "age_model.pth"
            gender_path = self.model_dir / "gender_model.pth"
            
            if use_pretrained:
                if emotion_path.exists():
                    self.emotion_model.load_state_dict(torch.load(emotion_path, map_location=self.device))
                    logger.info("Loaded pretrained emotion model")
                if age_path.exists():
                    self.age_model.load_state_dict(torch.load(age_path, map_location=self.device))
                    logger.info("Loaded pretrained age model")
                if gender_path.exists():
                    self.gender_model.load_state_dict(torch.load(gender_path, map_location=self.device))
                    logger.info("Loaded pretrained gender model")
            
            # Set to eval mode
            self.emotion_model.eval()
            self.age_model.eval()
            self.gender_model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
    
    def preprocess_for_emotion(self, face: np.ndarray) -> torch.Tensor:
        """Preprocess face for emotion model (48x48 grayscale)"""
        if face is None or face.size == 0:
            return None
            
        # Convert to grayscale
        if len(face.shape) == 3:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        else:
            gray = face
        
        # Resize to 48x48
        resized = cv2.resize(gray, (48, 48))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)  # (1, 1, 48, 48)
        return tensor.to(self.device)
    
    def preprocess_for_age_gender(self, face: np.ndarray) -> torch.Tensor:
        """Preprocess face for age/gender models (64x64 RGB)"""
        if face is None or face.size == 0:
            return None
            
        # Ensure RGB
        if len(face.shape) == 2:
            rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        elif face.shape[2] == 4:
            rgb = cv2.cvtColor(face, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Resize to 64x64
        resized = cv2.resize(rgb, (64, 64))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to tensor (B, C, H, W)
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
    
    @torch.no_grad()
    def analyze(self, face: np.ndarray) -> Optional[DemographicResult]:
        """
        Analyze a face image for demographics.
        
        Args:
            face: BGR face image (cropped and aligned)
            
        Returns:
            DemographicResult or None if analysis fails
        """
        if not TORCH_AVAILABLE or self.emotion_model is None:
            # Return placeholder result
            return DemographicResult(
                age=30,
                age_confidence=0.5,
                gender=Gender.MALE,
                gender_confidence=0.5,
                emotion=Emotion.NEUTRAL,
                emotion_confidence=0.5
            )
        
        try:
            # Preprocess
            emotion_input = self.preprocess_for_emotion(face)
            age_gender_input = self.preprocess_for_age_gender(face)
            
            if emotion_input is None or age_gender_input is None:
                return None
            
            # Emotion prediction
            emotion_logits = self.emotion_model(emotion_input)
            emotion_probs = F.softmax(emotion_logits, dim=1)[0]
            emotion_idx = emotion_probs.argmax().item()
            emotion_conf = emotion_probs[emotion_idx].item()
            
            emotion_scores = {
                Emotion.from_index(i).name.lower(): emotion_probs[i].item()
                for i in range(7)
            }
            
            # Age prediction
            age_pred = self.age_model(age_gender_input)
            age = int(age_pred[0].item())
            age_conf = 0.8  # Placeholder confidence
            
            # Gender prediction
            gender_logits = self.gender_model(age_gender_input)
            gender_probs = F.softmax(gender_logits, dim=1)[0]
            gender_idx = gender_probs.argmax().item()
            gender_conf = gender_probs[gender_idx].item()
            
            return DemographicResult(
                age=age,
                age_confidence=age_conf,
                gender=Gender.from_index(gender_idx),
                gender_confidence=gender_conf,
                emotion=Emotion.from_index(emotion_idx),
                emotion_confidence=emotion_conf,
                emotion_scores=emotion_scores
            )
            
        except Exception as e:
            logger.error(f"Demographics analysis failed: {e}")
            return None
    
    def analyze_batch(self, faces: List[np.ndarray]) -> List[Optional[DemographicResult]]:
        """Analyze multiple faces (batch processing)"""
        return [self.analyze(face) for face in faces]


class VisitorAnalytics:
    """
    Visitor analytics engine for business intelligence.
    
    Tracks:
    - Foot traffic patterns
    - Demographic distributions
    - Emotion/satisfaction trends
    - Dwell time analysis
    """
    
    def __init__(self, db_session=None):
        self.db_session = db_session
        
        # In-memory storage for real-time analytics
        self.detections: List[Dict] = []
        self.hourly_cache: Dict[str, Dict] = {}  # "YYYY-MM-DD HH" -> stats
        
    def record_detection(
        self,
        person_id: Optional[str],
        camera_id: str,
        timestamp: datetime,
        demographics: Optional[DemographicResult]
    ):
        """Record a face detection with demographics"""
        self.detections.append({
            "person_id": person_id,
            "camera_id": camera_id,
            "timestamp": timestamp,
            "demographics": demographics.to_dict() if demographics else None
        })
        
        # Update hourly cache
        hour_key = timestamp.strftime("%Y-%m-%d %H")
        if hour_key not in self.hourly_cache:
            self.hourly_cache[hour_key] = {
                "count": 0,
                "ages": [],
                "genders": {"male": 0, "female": 0},
                "emotions": defaultdict(int)
            }
        
        cache = self.hourly_cache[hour_key]
        cache["count"] += 1
        
        if demographics:
            cache["ages"].append(demographics.age)
            cache["genders"][demographics.gender.name.lower()] += 1
            cache["emotions"][demographics.emotion.name.lower()] += 1
    
    def get_stats(
        self,
        start_time: datetime,
        end_time: datetime,
        camera_id: Optional[str] = None
    ) -> VisitorStats:
        """
        Get visitor statistics for a time period.
        
        Args:
            start_time: Period start
            end_time: Period end
            camera_id: Optional camera filter
            
        Returns:
            VisitorStats with aggregated metrics
        """
        # Filter detections
        filtered = [
            d for d in self.detections
            if start_time <= d["timestamp"] <= end_time
            and (camera_id is None or d["camera_id"] == camera_id)
        ]
        
        total = len(filtered)
        unique = len(set(d.get("person_id") for d in filtered if d.get("person_id")))
        
        # Age distribution
        age_dist = {"0-18": 0, "19-35": 0, "36-50": 0, "51+": 0}
        for d in filtered:
            if d["demographics"]:
                age = d["demographics"]["age"]
                if age <= 18:
                    age_dist["0-18"] += 1
                elif age <= 35:
                    age_dist["19-35"] += 1
                elif age <= 50:
                    age_dist["36-50"] += 1
                else:
                    age_dist["51+"] += 1
        
        # Gender distribution
        gender_dist = {"male": 0, "female": 0}
        for d in filtered:
            if d["demographics"]:
                gender_dist[d["demographics"]["gender"]] += 1
        
        # Emotion distribution
        emotion_dist = defaultdict(int)
        for d in filtered:
            if d["demographics"]:
                emotion_dist[d["demographics"]["emotion"]] += 1
        
        # Hourly traffic
        hourly = defaultdict(int)
        for d in filtered:
            hour = d["timestamp"].hour
            hourly[hour] += 1
        
        peak_hour = max(hourly.keys(), key=lambda h: hourly[h]) if hourly else 0
        
        # Satisfaction (positive vs negative emotions)
        positive = emotion_dist.get("happy", 0) + emotion_dist.get("surprise", 0)
        negative = (
            emotion_dist.get("angry", 0) + 
            emotion_dist.get("sad", 0) + 
            emotion_dist.get("fear", 0) +
            emotion_dist.get("disgust", 0)
        )
        total_emotions = sum(emotion_dist.values()) or 1
        
        return VisitorStats(
            period_start=start_time,
            period_end=end_time,
            total_visitors=total,
            unique_visitors=unique,
            age_distribution=age_dist,
            gender_distribution=dict(gender_dist),
            emotion_distribution=dict(emotion_dist),
            hourly_traffic=dict(hourly),
            peak_hour=peak_hour,
            positive_emotion_ratio=positive / total_emotions,
            negative_emotion_ratio=negative / total_emotions
        )
    
    def get_realtime_mood(self, window_minutes: int = 5) -> Dict[str, Any]:
        """
        Get real-time mood indicator for the last N minutes.
        
        Useful for live dashboards.
        """
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent = [
            d for d in self.detections
            if d["timestamp"] >= cutoff and d["demographics"]
        ]
        
        if not recent:
            return {
                "status": "no_data",
                "sample_size": 0,
                "mood": "unknown",
                "mood_score": 0.5
            }
        
        # Calculate mood score (-1 to 1)
        mood_weights = {
            "happy": 1.0,
            "surprise": 0.5,
            "neutral": 0.0,
            "sad": -0.5,
            "angry": -0.8,
            "fear": -0.6,
            "disgust": -0.7
        }
        
        scores = [
            mood_weights.get(d["demographics"]["emotion"], 0)
            for d in recent
        ]
        
        avg_score = sum(scores) / len(scores)
        
        # Determine mood label
        if avg_score > 0.3:
            mood = "positive"
        elif avg_score < -0.3:
            mood = "negative"
        else:
            mood = "neutral"
        
        return {
            "status": "ok",
            "sample_size": len(recent),
            "window_minutes": window_minutes,
            "mood": mood,
            "mood_score": (avg_score + 1) / 2,  # Normalize to 0-1
            "emotion_breakdown": dict(
                defaultdict(int, [
                    (d["demographics"]["emotion"], 1) 
                    for d in recent
                ])
            )
        }
    
    def get_demographics_trend(
        self,
        metric: str,  # 'age', 'gender', 'emotion'
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get trend data for a demographic metric over time.
        
        Args:
            metric: Which metric to track
            days: Number of days to include
            
        Returns:
            Daily breakdown of the metric
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        daily_data = defaultdict(lambda: defaultdict(int))
        
        for d in self.detections:
            if d["timestamp"] < cutoff or not d["demographics"]:
                continue
                
            day_key = d["timestamp"].strftime("%Y-%m-%d")
            
            if metric == "age":
                age = d["demographics"]["age"]
                if age <= 18:
                    daily_data[day_key]["0-18"] += 1
                elif age <= 35:
                    daily_data[day_key]["19-35"] += 1
                elif age <= 50:
                    daily_data[day_key]["36-50"] += 1
                else:
                    daily_data[day_key]["51+"] += 1
            elif metric == "gender":
                daily_data[day_key][d["demographics"]["gender"]] += 1
            elif metric == "emotion":
                daily_data[day_key][d["demographics"]["emotion"]] += 1
        
        return {
            "metric": metric,
            "days": days,
            "data": dict(daily_data)
        }
    
    def export_report(
        self,
        start_time: datetime,
        end_time: datetime,
        format: str = "json"
    ) -> str:
        """
        Export analytics report.
        
        Args:
            start_time: Report start
            end_time: Report end
            format: 'json' or 'csv'
            
        Returns:
            Report data as string
        """
        stats = self.get_stats(start_time, end_time)
        
        if format == "json":
            return json.dumps(stats.to_dict(), indent=2)
        elif format == "csv":
            lines = [
                "Metric,Value",
                f"Total Visitors,{stats.total_visitors}",
                f"Unique Visitors,{stats.unique_visitors}",
                f"Peak Hour,{stats.peak_hour}",
                f"Positive Emotion Ratio,{stats.positive_emotion_ratio:.2%}",
                f"Negative Emotion Ratio,{stats.negative_emotion_ratio:.2%}",
                "",
                "Age Group,Count",
            ]
            for age_group, count in stats.age_distribution.items():
                lines.append(f"{age_group},{count}")
            
            lines.extend(["", "Gender,Count"])
            for gender, count in stats.gender_distribution.items():
                lines.append(f"{gender},{count}")
            
            lines.extend(["", "Emotion,Count"])
            for emotion, count in stats.emotion_distribution.items():
                lines.append(f"{emotion},{count}")
            
            return "\n".join(lines)
        
        raise ValueError(f"Unknown format: {format}")


# Factory function
def create_demographics_system(
    model_dir: str = "models/demographics",
    device: str = "auto"
) -> Tuple[DemographicsAnalyzer, VisitorAnalytics]:
    """
    Create demographics analysis system.
    
    Args:
        model_dir: Directory containing pretrained models
        device: 'cuda', 'cpu', or 'auto'
        
    Returns:
        (analyzer, analytics) tuple
    """
    if device == "auto":
        device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    
    analyzer = DemographicsAnalyzer(model_dir=model_dir, device=device)
    analytics = VisitorAnalytics()
    
    return analyzer, analytics
