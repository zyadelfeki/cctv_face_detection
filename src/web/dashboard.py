"""
Real-time CCTV Face Detection Dashboard
Built with Streamlit for live monitoring and analytics
"""

import streamlit as st
import cv2
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import threading
import time
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass, field
import queue


@dataclass
class DetectionEvent:
    """Represents a face detection event."""
    timestamp: datetime
    camera_id: str
    person_id: Optional[str]
    person_name: Optional[str]
    confidence: float
    is_known: bool
    bbox: Tuple[int, int, int, int]
    embedding: Optional[np.ndarray] = None
    snapshot_path: Optional[str] = None


@dataclass
class CameraStats:
    """Statistics for a single camera."""
    camera_id: str
    total_detections: int = 0
    known_faces: int = 0
    unknown_faces: int = 0
    alerts_triggered: int = 0
    last_detection: Optional[datetime] = None
    fps: float = 0.0
    status: str = "offline"


class DashboardState:
    """Global state manager for the dashboard."""
    
    def __init__(self):
        self.detection_history: deque = deque(maxlen=1000)
        self.camera_stats: Dict[str, CameraStats] = {}
        self.alerts: deque = deque(maxlen=100)
        self.hourly_stats: Dict[str, List[int]] = {}
        self.frame_queues: Dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
    
    def add_detection(self, event: DetectionEvent):
        """Add a detection event."""
        with self._lock:
            self.detection_history.append(event)
            
            # Update camera stats
            if event.camera_id not in self.camera_stats:
                self.camera_stats[event.camera_id] = CameraStats(camera_id=event.camera_id)
            
            stats = self.camera_stats[event.camera_id]
            stats.total_detections += 1
            stats.last_detection = event.timestamp
            
            if event.is_known:
                stats.known_faces += 1
            else:
                stats.unknown_faces += 1
    
    def add_alert(self, alert: Dict):
        """Add an alert."""
        with self._lock:
            self.alerts.appendleft(alert)
            if alert.get('camera_id') in self.camera_stats:
                self.camera_stats[alert['camera_id']].alerts_triggered += 1
    
    def get_recent_detections(self, minutes: int = 60) -> List[DetectionEvent]:
        """Get detections from the last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        with self._lock:
            return [d for d in self.detection_history if d.timestamp > cutoff]
    
    def get_hourly_breakdown(self) -> Dict[int, int]:
        """Get detection counts by hour."""
        hourly = {h: 0 for h in range(24)}
        for event in self.detection_history:
            hourly[event.timestamp.hour] += 1
        return hourly


# Initialize global state
if 'dashboard_state' not in st.session_state:
    st.session_state.dashboard_state = DashboardState()


def render_header():
    """Render the dashboard header."""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.title("🎥 CCTV Face Detection Dashboard")
    
    with col2:
        st.metric("System Status", "🟢 Online")
    
    with col3:
        st.metric("Active Cameras", len(st.session_state.dashboard_state.camera_stats))


def render_live_feeds():
    """Render live camera feeds section."""
    st.subheader("📹 Live Camera Feeds")
    
    # Demo: Create placeholder feeds
    num_cameras = st.slider("Number of cameras to display", 1, 6, 4)
    cols = st.columns(min(num_cameras, 3))
    
    for i in range(num_cameras):
        with cols[i % 3]:
            st.markdown(f"**Camera {i + 1}**")
            
            # Placeholder for video feed
            # In production, this would be replaced with actual RTSP stream
            placeholder = st.empty()
            
            # Demo stats
            stats = st.session_state.dashboard_state.camera_stats.get(
                f"cam_{i+1}", 
                CameraStats(camera_id=f"cam_{i+1}")
            )
            
            status_color = "🟢" if stats.status == "online" else "🔴"
            st.caption(f"{status_color} Status: {stats.status} | FPS: {stats.fps:.1f}")


def render_statistics():
    """Render real-time statistics."""
    st.subheader("📊 Real-time Statistics")
    
    state = st.session_state.dashboard_state
    recent = state.get_recent_detections(60)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total = len(recent)
        st.metric(
            "Detections (Last Hour)", 
            total,
            delta=f"+{len(state.get_recent_detections(5))} (5min)"
        )
    
    with col2:
        known = sum(1 for d in recent if d.is_known)
        st.metric("Known Faces", known)
    
    with col3:
        unknown = sum(1 for d in recent if not d.is_known)
        st.metric("Unknown Faces", unknown)
    
    with col4:
        alerts = len(state.alerts)
        st.metric("Active Alerts", alerts)


def render_hourly_chart():
    """Render hourly detection chart."""
    st.subheader("📈 Detection Timeline")
    
    state = st.session_state.dashboard_state
    hourly = state.get_hourly_breakdown()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(hourly.keys()),
        y=list(hourly.values()),
        marker_color='rgb(55, 83, 109)',
        name='Detections'
    ))
    
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Number of Detections",
        height=300,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_detection_map():
    """Render detection heatmap by camera."""
    st.subheader("🗺️ Camera Detection Heatmap")
    
    state = st.session_state.dashboard_state
    
    if state.camera_stats:
        cameras = list(state.camera_stats.keys())
        detections = [s.total_detections for s in state.camera_stats.values()]
        
        fig = px.bar(
            x=cameras,
            y=detections,
            color=detections,
            color_continuous_scale='Viridis',
            labels={'x': 'Camera', 'y': 'Detections'}
        )
        
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No camera data available yet.")


def render_alerts_panel():
    """Render the alerts panel."""
    st.subheader("🚨 Recent Alerts")
    
    state = st.session_state.dashboard_state
    
    if state.alerts:
        for alert in list(state.alerts)[:10]:
            severity_icon = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }.get(alert.get('severity', 'low'), '⚪')
            
            with st.expander(
                f"{severity_icon} {alert.get('title', 'Alert')} - {alert.get('time', 'Unknown time')}"
            ):
                st.write(alert.get('message', 'No details'))
                if alert.get('snapshot'):
                    st.image(alert['snapshot'], width=200)
    else:
        st.info("No recent alerts")


def render_recent_detections():
    """Render recent detection log."""
    st.subheader("👤 Recent Detections")
    
    state = st.session_state.dashboard_state
    recent = state.get_recent_detections(30)
    
    if recent:
        for det in list(recent)[-10:]:
            icon = "✅" if det.is_known else "❓"
            name = det.person_name or "Unknown"
            conf = f"{det.confidence:.1%}"
            
            st.markdown(
                f"{icon} **{name}** | Camera: {det.camera_id} | "
                f"Confidence: {conf} | {det.timestamp.strftime('%H:%M:%S')}"
            )
    else:
        st.info("No recent detections")


def render_camera_management():
    """Render camera management section."""
    st.subheader("⚙️ Camera Management")
    
    with st.expander("Add New Camera"):
        col1, col2 = st.columns(2)
        with col1:
            camera_name = st.text_input("Camera Name")
            camera_url = st.text_input("RTSP URL", placeholder="rtsp://...")
        with col2:
            camera_location = st.text_input("Location")
            camera_type = st.selectbox("Camera Type", ["Indoor", "Outdoor", "PTZ"])
        
        if st.button("Add Camera"):
            st.success(f"Camera '{camera_name}' added successfully!")
    
    with st.expander("Camera List"):
        state = st.session_state.dashboard_state
        
        if state.camera_stats:
            for cam_id, stats in state.camera_stats.items():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                with col1:
                    st.write(f"📷 {cam_id}")
                with col2:
                    st.write(f"Detections: {stats.total_detections}")
                with col3:
                    status = "🟢 Online" if stats.status == "online" else "🔴 Offline"
                    st.write(status)
                with col4:
                    st.button("Configure", key=f"cfg_{cam_id}")
        else:
            st.info("No cameras configured")


def render_person_database():
    """Render person database management."""
    st.subheader("👥 Known Persons Database")
    
    with st.expander("Add New Person"):
        col1, col2 = st.columns(2)
        with col1:
            person_name = st.text_input("Person Name")
            person_role = st.selectbox("Role", ["Employee", "VIP", "Visitor", "Restricted"])
        with col2:
            uploaded_file = st.file_uploader("Upload Photo", type=['jpg', 'png'])
        
        if st.button("Add Person"):
            if person_name and uploaded_file:
                st.success(f"Person '{person_name}' added to database!")
            else:
                st.warning("Please provide name and photo")
    
    # Demo persons list
    st.markdown("**Registered Persons:**")
    demo_persons = [
        {"name": "John Doe", "role": "Employee", "last_seen": "10 mins ago"},
        {"name": "Jane Smith", "role": "VIP", "last_seen": "2 hours ago"},
        {"name": "Bob Wilson", "role": "Employee", "last_seen": "Yesterday"},
    ]
    
    for person in demo_persons:
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.write(f"👤 {person['name']}")
        with col2:
            st.write(person['role'])
        with col3:
            st.write(person['last_seen'])
        with col4:
            st.button("View", key=f"view_{person['name']}")


def render_settings():
    """Render settings panel."""
    st.subheader("⚙️ Settings")
    
    with st.expander("Detection Settings"):
        st.slider("Detection Confidence Threshold", 0.0, 1.0, 0.8)
        st.slider("Recognition Confidence Threshold", 0.0, 1.0, 0.7)
        st.number_input("Min Face Size (pixels)", 20, 200, 40)
        st.checkbox("Enable Liveness Detection", value=True)
    
    with st.expander("Alert Settings"):
        st.multiselect(
            "Alert Channels",
            ["Email", "SMS", "Slack", "Discord", "Push Notification"],
            default=["Email"]
        )
        st.checkbox("Alert on Unknown Face", value=True)
        st.checkbox("Alert on Known Restricted Person", value=True)
        st.number_input("Alert Cooldown (seconds)", 0, 300, 60)
    
    with st.expander("Performance Settings"):
        st.selectbox("Processing Mode", ["Real-time", "Balanced", "Power Saver"])
        st.checkbox("Enable GPU Acceleration", value=True)
        st.number_input("Max Concurrent Cameras", 1, 20, 8)


def add_demo_data():
    """Add demo data for testing."""
    import random
    
    state = st.session_state.dashboard_state
    
    # Add demo cameras
    for i in range(4):
        cam_id = f"cam_{i+1}"
        if cam_id not in state.camera_stats:
            state.camera_stats[cam_id] = CameraStats(
                camera_id=cam_id,
                total_detections=random.randint(50, 200),
                known_faces=random.randint(20, 100),
                unknown_faces=random.randint(10, 50),
                fps=random.uniform(25, 30),
                status="online"
            )
    
    # Add demo detections
    names = ["John Doe", "Jane Smith", None, "Bob Wilson", None, None]
    for i in range(20):
        name = random.choice(names)
        event = DetectionEvent(
            timestamp=datetime.now() - timedelta(minutes=random.randint(0, 60)),
            camera_id=f"cam_{random.randint(1, 4)}",
            person_id=f"person_{i}" if name else None,
            person_name=name,
            confidence=random.uniform(0.7, 0.99),
            is_known=name is not None,
            bbox=(100, 100, 200, 200)
        )
        state.detection_history.append(event)
    
    # Add demo alerts
    alert_types = [
        {"title": "Unknown person detected", "severity": "medium"},
        {"title": "Restricted area access", "severity": "high"},
        {"title": "Multiple faces detected", "severity": "low"},
    ]
    for i in range(5):
        alert = random.choice(alert_types).copy()
        alert['time'] = (datetime.now() - timedelta(minutes=random.randint(0, 30))).strftime('%H:%M:%S')
        alert['message'] = f"Detection event on Camera {random.randint(1, 4)}"
        alert['camera_id'] = f"cam_{random.randint(1, 4)}"
        state.alerts.append(alert)


def main():
    """Main dashboard entry point."""
    st.set_page_config(
        page_title="CCTV Face Detection Dashboard",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
        }
        .stAlert {
            padding: 10px;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "Cameras", "Persons", "Alerts", "Settings"]
    )
    
    # Add demo data button
    if st.sidebar.button("Load Demo Data"):
        add_demo_data()
        st.sidebar.success("Demo data loaded!")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False)
    if auto_refresh:
        time.sleep(5)
        st.rerun()
    
    # Render page based on selection
    if page == "Dashboard":
        render_header()
        render_statistics()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            render_hourly_chart()
            render_live_feeds()
        with col2:
            render_alerts_panel()
            render_recent_detections()
    
    elif page == "Cameras":
        st.title("📷 Camera Management")
        render_camera_management()
        render_detection_map()
    
    elif page == "Persons":
        st.title("👥 Person Database")
        render_person_database()
    
    elif page == "Alerts":
        st.title("🚨 Alert Management")
        render_alerts_panel()
        render_recent_detections()
    
    elif page == "Settings":
        st.title("⚙️ System Settings")
        render_settings()


if __name__ == "__main__":
    main()
