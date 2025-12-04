# 🎓 CCTV Face Detection System - Graduation Project Features

## 📋 Project Overview

A comprehensive AI-powered CCTV surveillance system with advanced face detection, recognition, and analytics capabilities. This project showcases state-of-the-art computer vision techniques suitable for a graduation project presentation.

---

## ✅ Implemented Features

### 1. 🖥️ Real-time Dashboard (`src/web/dashboard.py`)
**Technology:** Streamlit + Plotly + WebSocket

- **Live Camera Grid**: View all cameras simultaneously
- **Real-time Statistics**: Detection counts, recognition accuracy
- **Alert Panel**: Instant notifications for security events
- **Detection History**: Searchable log of all detections
- **Interactive Charts**: Hourly/daily trends visualization

**Launch:** `streamlit run run_dashboard.py`

---

### 2. 🔒 Enhanced Liveness Detection (`src/core/liveness/advanced.py`)
**Technology:** MiDaS + OpenCV + MediaPipe

4-Layer Anti-Spoofing Protection:
1. **Depth Estimation**: MiDaS neural network detects flat surfaces (photos/screens)
2. **Challenge-Response**: Random head movements (turn left, look up, blink)
3. **3D Face Reconstruction**: PnP pose estimation validates 3D geometry
4. **Blink Detection**: Eye Aspect Ratio (EAR) algorithm for liveness

---

### 3. 👥 Face Clustering (`src/core/clustering.py`)
**Technology:** HDBSCAN + Cosine Similarity

- **Unknown Face Grouping**: Automatically cluster unidentified faces
- **Person of Interest Alerts**: Flag frequently appearing unknowns
- **Cluster Statistics**: Track appearance frequency, first/last seen
- **Re-clustering**: Merge clusters when identity is confirmed

---

### 4. 🔧 Edge Deployment (`src/core/edge_deployment.py`)
**Technology:** ONNX + TensorRT + OpenVINO

**Optimization Targets:**
- **ONNX Export**: Universal model format
- **TensorRT**: NVIDIA GPU optimization (FP16/INT8)
- **OpenVINO**: Intel CPU/GPU optimization
- **TensorFlow Lite**: Raspberry Pi deployment

**Supported Devices:**
- NVIDIA Jetson (Nano, Xavier, Orin)
- Raspberry Pi 4/5
- Intel NCS2

---

### 5. 📹 Multi-Camera Tracking (`src/core/multi_camera_tracker.py`)
**Technology:** OSNet Re-ID + Hungarian Algorithm

- **Cross-Camera Re-ID**: Track same person across different cameras
- **Movement Timeline**: Visualize person's path through facility
- **Track Lifecycle**: Active → Lost → Archived states
- **Appearance Fusion**: Combine face + body features

---

### 6. 📱 Mobile App Integration (`src/core/mobile_integration.py`)
**Technology:** Firebase Cloud Messaging + WebSocket

**Features:**
- **Push Notifications**: Face detected, unknown person, security alerts
- **Device Management**: Register multiple devices per user
- **Adaptive Streaming**: Adjusts quality based on network (WiFi/4G/3G)
- **Offline Sync**: Queue events when device offline

**API Endpoints:** `src/api/mobile_routes.py`

---

### 7. 🔍 Searchable Face Database (`src/core/face_database.py`)
**Technology:** FAISS + SQLite

**Search Capabilities:**
- **Photo Search**: Upload image → find all appearances
- **Time-Range Queries**: "Show faces from 9am-5pm yesterday"
- **Person Timeline**: Track movement through cameras
- **Multi-Filter**: Combine camera, time, confidence, demographics

**Export Options:**
- Snapshots with bounding boxes
- Video clips (±5 seconds around detection)
- ZIP archives for bulk export

**API Endpoints:** `src/api/search_routes.py`

---

### 8. 😊 Emotion & Demographics (`src/core/demographics.py`)
**Technology:** Mini-Xception + SSR-Net

**Analysis:**
- **Age Estimation**: SSR-Net regression (0-100 years)
- **Gender Classification**: Binary classification
- **Emotion Recognition**: 7 emotions (happy, sad, angry, fear, surprise, disgust, neutral)

**Business Intelligence:**
- Visitor statistics (hourly traffic, peak hours)
- Demographic distributions (age groups, gender)
- Real-time mood indicator
- Satisfaction metrics (positive/negative emotion ratio)
- Period comparison (week-over-week)

**API Endpoints:** `src/api/analytics_routes.py`

---

## 📁 Project Structure

```
cctv_face_detection/
├── src/
│   ├── core/
│   │   ├── liveness/
│   │   │   ├── detector.py          # Base liveness detection
│   │   │   └── advanced.py          # Depth, 3D, challenge-response
│   │   ├── clustering.py            # HDBSCAN face clustering
│   │   ├── edge_deployment.py       # ONNX/TensorRT optimization
│   │   ├── multi_camera_tracker.py  # Cross-camera Re-ID
│   │   ├── mobile_integration.py    # Firebase push notifications
│   │   ├── face_database.py         # FAISS vector search
│   │   └── demographics.py          # Age/gender/emotion
│   ├── api/
│   │   ├── dashboard_routes.py      # WebSocket + REST for dashboard
│   │   ├── mobile_routes.py         # Mobile app endpoints
│   │   ├── search_routes.py         # Face search endpoints
│   │   └── analytics_routes.py      # Demographics endpoints
│   └── web/
│       ├── dashboard.py             # Streamlit UI
│       └── stream_manager.py        # Camera stream handling
├── notebooks/
│   └── face_recognition_training.ipynb  # Colab training notebook
├── run_dashboard.py                 # Dashboard launcher
└── requirements.txt                 # All dependencies
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start API Server
```bash
uvicorn src.api.main:app --reload
```

### 3. Launch Dashboard
```bash
streamlit run run_dashboard.py
```

### 4. Access Endpoints
- Dashboard: http://localhost:8501
- API Docs: http://localhost:8000/docs
- WebSocket: ws://localhost:8000/ws/dashboard

---

## 📊 API Endpoints Summary

| Category | Endpoint | Description |
|----------|----------|-------------|
| **Dashboard** | `GET /api/dashboard/stats` | Real-time statistics |
| | `WS /ws/dashboard` | Live updates |
| **Mobile** | `POST /api/mobile/devices/register` | Register device |
| | `POST /api/mobile/test/notification` | Test push |
| **Search** | `POST /api/faces/search` | Multi-filter search |
| | `POST /api/faces/search/photo` | Photo upload search |
| | `GET /api/faces/timeline/{person_id}` | Movement timeline |
| **Analytics** | `POST /api/analytics/analyze` | Demographic analysis |
| | `GET /api/analytics/mood` | Real-time mood |
| | `GET /api/analytics/dashboard` | Full dashboard data |

---

## 🎯 Technologies Demonstrated

1. **Deep Learning**: CNN, ResNet, MTCNN, MiDaS, Mini-Xception
2. **Computer Vision**: OpenCV, face detection, landmark detection
3. **Vector Search**: FAISS billion-scale similarity search
4. **Real-time Systems**: WebSocket, streaming, async Python
5. **Edge Computing**: ONNX, TensorRT, model quantization
6. **Mobile Integration**: Firebase, push notifications
7. **Web Development**: FastAPI, Streamlit, REST APIs
8. **Data Analysis**: Pandas, Plotly, time-series analytics

---

## 📝 License

MIT License - Feel free to use for educational purposes.

---

**Author**: Graduation Project 2024
