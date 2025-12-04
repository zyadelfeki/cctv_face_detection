"""
Demographics & Analytics API Routes

REST endpoints for emotion and demographics analysis:
- Real-time demographic analysis
- Visitor statistics
- Mood indicators
- Trend analysis

Author: CCTV Face Detection System
"""

import asyncio
import io
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
import numpy as np
import cv2

from ..core.demographics import (
    DemographicsAnalyzer,
    DemographicResult,
    VisitorAnalytics,
    VisitorStats,
    create_demographics_system
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analytics", tags=["demographics"])

# Global instances
demographics_analyzer: Optional[DemographicsAnalyzer] = None
visitor_analytics: Optional[VisitorAnalytics] = None


# ============ Pydantic Models ============

class DemographicResponse(BaseModel):
    """Response for demographic analysis"""
    age: int
    age_confidence: float
    gender: str
    gender_confidence: float
    emotion: str
    emotion_confidence: float
    emotion_scores: Dict[str, float]


class VisitorStatsResponse(BaseModel):
    """Response for visitor statistics"""
    period_start: datetime
    period_end: datetime
    total_visitors: int
    unique_visitors: int
    age_distribution: Dict[str, int]
    gender_distribution: Dict[str, int]
    emotion_distribution: Dict[str, int]
    hourly_traffic: Dict[int, int]
    peak_hour: int
    positive_emotion_ratio: float
    negative_emotion_ratio: float


class MoodResponse(BaseModel):
    """Response for real-time mood"""
    status: str
    sample_size: int
    window_minutes: int
    mood: str
    mood_score: float
    emotion_breakdown: Dict[str, int]


class TrendResponse(BaseModel):
    """Response for trend analysis"""
    metric: str
    days: int
    data: Dict[str, Dict[str, int]]


class HealthResponse(BaseModel):
    """Service health check"""
    status: str
    models_loaded: bool
    device: str
    total_detections: int


# ============ Lifecycle ============

async def init_demographics_services(
    model_dir: str = "models/demographics",
    device: str = "auto"
):
    """Initialize demographics services on startup"""
    global demographics_analyzer, visitor_analytics
    
    demographics_analyzer, visitor_analytics = create_demographics_system(
        model_dir=model_dir,
        device=device
    )
    
    logger.info("Demographics services initialized")


# ============ Analysis Endpoints ============

@router.post("/analyze", response_model=DemographicResponse)
async def analyze_face(
    image: UploadFile = File(..., description="Face image to analyze")
):
    """
    Analyze a face image for demographics.
    
    Returns age, gender, and emotion predictions.
    """
    if not demographics_analyzer:
        raise HTTPException(status_code=503, detail="Demographics service not initialized")
    
    # Read image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Analyze
    result = demographics_analyzer.analyze(img)
    
    if result is None:
        raise HTTPException(status_code=400, detail="Face analysis failed")
    
    return DemographicResponse(
        age=result.age,
        age_confidence=result.age_confidence,
        gender=result.gender.name.lower(),
        gender_confidence=result.gender_confidence,
        emotion=result.emotion.name.lower(),
        emotion_confidence=result.emotion_confidence,
        emotion_scores=result.emotion_scores
    )


@router.post("/analyze/batch")
async def analyze_faces_batch(
    images: List[UploadFile] = File(..., description="Multiple face images")
):
    """
    Analyze multiple face images.
    
    More efficient for bulk processing.
    """
    if not demographics_analyzer:
        raise HTTPException(status_code=503, detail="Demographics service not initialized")
    
    results = []
    
    for image in images:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is not None:
            result = demographics_analyzer.analyze(img)
            if result:
                results.append({
                    "filename": image.filename,
                    "result": result.to_dict()
                })
            else:
                results.append({
                    "filename": image.filename,
                    "error": "Analysis failed"
                })
        else:
            results.append({
                "filename": image.filename,
                "error": "Invalid image"
            })
    
    return {
        "total": len(images),
        "successful": len([r for r in results if "result" in r]),
        "results": results
    }


# ============ Statistics Endpoints ============

@router.get("/stats", response_model=VisitorStatsResponse)
async def get_visitor_stats(
    start_time: datetime = Query(..., description="Start of period"),
    end_time: datetime = Query(..., description="End of period"),
    camera_id: Optional[str] = Query(default=None, description="Filter by camera")
):
    """
    Get visitor statistics for a time period.
    
    Includes demographic breakdowns, traffic patterns, and satisfaction metrics.
    """
    if not visitor_analytics:
        raise HTTPException(status_code=503, detail="Analytics service not initialized")
    
    stats = visitor_analytics.get_stats(start_time, end_time, camera_id)
    
    return VisitorStatsResponse(
        period_start=stats.period_start,
        period_end=stats.period_end,
        total_visitors=stats.total_visitors,
        unique_visitors=stats.unique_visitors,
        age_distribution=stats.age_distribution,
        gender_distribution=stats.gender_distribution,
        emotion_distribution=stats.emotion_distribution,
        hourly_traffic=stats.hourly_traffic,
        peak_hour=stats.peak_hour,
        positive_emotion_ratio=stats.positive_emotion_ratio,
        negative_emotion_ratio=stats.negative_emotion_ratio
    )


@router.get("/stats/today")
async def get_today_stats(
    camera_id: Optional[str] = Query(default=None)
):
    """Get today's visitor statistics"""
    if not visitor_analytics:
        raise HTTPException(status_code=503, detail="Analytics service not initialized")
    
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)
    
    stats = visitor_analytics.get_stats(today, tomorrow, camera_id)
    return stats.to_dict()


@router.get("/stats/week")
async def get_week_stats(
    camera_id: Optional[str] = Query(default=None)
):
    """Get this week's visitor statistics"""
    if not visitor_analytics:
        raise HTTPException(status_code=503, detail="Analytics service not initialized")
    
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago = today - timedelta(days=7)
    
    stats = visitor_analytics.get_stats(week_ago, today, camera_id)
    return stats.to_dict()


# ============ Real-time Endpoints ============

@router.get("/mood", response_model=MoodResponse)
async def get_realtime_mood(
    window_minutes: int = Query(default=5, ge=1, le=60, description="Analysis window")
):
    """
    Get real-time mood indicator.
    
    Analyzes emotions from recent detections to determine
    overall visitor mood (positive/neutral/negative).
    
    Useful for:
    - Retail: Monitor customer satisfaction
    - Security: Detect potential issues
    - Events: Gauge crowd sentiment
    """
    if not visitor_analytics:
        raise HTTPException(status_code=503, detail="Analytics service not initialized")
    
    mood_data = visitor_analytics.get_realtime_mood(window_minutes)
    
    return MoodResponse(
        status=mood_data["status"],
        sample_size=mood_data.get("sample_size", 0),
        window_minutes=window_minutes,
        mood=mood_data.get("mood", "unknown"),
        mood_score=mood_data.get("mood_score", 0.5),
        emotion_breakdown=mood_data.get("emotion_breakdown", {})
    )


@router.get("/trend/{metric}", response_model=TrendResponse)
async def get_demographic_trend(
    metric: str,
    days: int = Query(default=7, ge=1, le=90)
):
    """
    Get trend data for a demographic metric.
    
    Metrics:
    - `age`: Age group distribution over time
    - `gender`: Gender distribution over time
    - `emotion`: Emotion distribution over time
    """
    if not visitor_analytics:
        raise HTTPException(status_code=503, detail="Analytics service not initialized")
    
    if metric not in ["age", "gender", "emotion"]:
        raise HTTPException(status_code=400, detail="Invalid metric. Use: age, gender, emotion")
    
    trend = visitor_analytics.get_demographics_trend(metric, days)
    
    return TrendResponse(
        metric=trend["metric"],
        days=trend["days"],
        data=trend["data"]
    )


# ============ Report Endpoints ============

@router.get("/report")
async def get_analytics_report(
    start_time: datetime = Query(...),
    end_time: datetime = Query(...),
    format: str = Query(default="json", regex="^(json|csv)$")
):
    """
    Export analytics report.
    
    Formats:
    - `json`: Structured JSON data
    - `csv`: CSV for spreadsheet import
    """
    if not visitor_analytics:
        raise HTTPException(status_code=503, detail="Analytics service not initialized")
    
    report = visitor_analytics.export_report(start_time, end_time, format)
    
    if format == "csv":
        return PlainTextResponse(
            content=report,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=analytics_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"
            }
        )
    
    return {"report": report}


# ============ Dashboard Data ============

@router.get("/dashboard")
async def get_dashboard_data():
    """
    Get all data needed for analytics dashboard.
    
    Single endpoint for populating dashboard widgets.
    """
    if not visitor_analytics:
        raise HTTPException(status_code=503, detail="Analytics service not initialized")
    
    now = datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago = today - timedelta(days=7)
    
    return {
        "timestamp": now.isoformat(),
        "today": visitor_analytics.get_stats(today, now).to_dict(),
        "this_week": visitor_analytics.get_stats(week_ago, now).to_dict(),
        "realtime_mood": visitor_analytics.get_realtime_mood(5),
        "age_trend": visitor_analytics.get_demographics_trend("age", 7),
        "emotion_trend": visitor_analytics.get_demographics_trend("emotion", 7)
    }


# ============ Comparison Endpoints ============

@router.get("/compare")
async def compare_periods(
    period1_start: datetime = Query(...),
    period1_end: datetime = Query(...),
    period2_start: datetime = Query(...),
    period2_end: datetime = Query(...),
    camera_id: Optional[str] = None
):
    """
    Compare visitor statistics between two periods.
    
    Useful for:
    - Week-over-week comparison
    - Before/after campaign analysis
    - Seasonal comparisons
    """
    if not visitor_analytics:
        raise HTTPException(status_code=503, detail="Analytics service not initialized")
    
    stats1 = visitor_analytics.get_stats(period1_start, period1_end, camera_id)
    stats2 = visitor_analytics.get_stats(period2_start, period2_end, camera_id)
    
    def calc_change(v1: float, v2: float) -> Dict:
        if v1 == 0:
            return {"value": v2, "change": None, "percent_change": None}
        change = v2 - v1
        pct = (change / v1) * 100
        return {"value": v2, "change": change, "percent_change": round(pct, 1)}
    
    return {
        "period1": {
            "start": period1_start.isoformat(),
            "end": period1_end.isoformat(),
            "stats": stats1.to_dict()
        },
        "period2": {
            "start": period2_start.isoformat(),
            "end": period2_end.isoformat(),
            "stats": stats2.to_dict()
        },
        "comparison": {
            "total_visitors": calc_change(stats1.total_visitors, stats2.total_visitors),
            "unique_visitors": calc_change(stats1.unique_visitors, stats2.unique_visitors),
            "positive_ratio": calc_change(stats1.positive_emotion_ratio, stats2.positive_emotion_ratio)
        }
    }


# ============ Health ============

@router.get("/health", response_model=HealthResponse)
async def demographics_health():
    """Health check for demographics services"""
    return HealthResponse(
        status="ok" if demographics_analyzer else "degraded",
        models_loaded=demographics_analyzer is not None,
        device=demographics_analyzer.device if demographics_analyzer else "none",
        total_detections=len(visitor_analytics.detections) if visitor_analytics else 0
    )
