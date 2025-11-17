# Deep AI CCTV System — How & Why Everything Works

## What is This Project?
This repo is an all-in-one AI face detection and recognition system for real-time CCTV monitoring — from raw video stream, to real criminal match, to instant alert. We built this to: 
- Show off actual deep learning skills (not just gluing libraries!)
- Be easy enough for anyone (professors, devs, random noobs) to run or debug
- Let you **see the entire ML/AI pipeline in action**

---

## 1. The AI Pipeline: From Camera to Alert

### Step by Step (with real explanation):

1. **Video Frame Ingestion:**
   - Takes live video from any RTSP/IP cam (think: security cam). Converts each video frame into an image you can run AI on.
   - Handles drops and reconnects, so if the camera disconnects, the system retries automatically.

2. **MTCNN Face Detection:**
   - Every frame, the system runs an MTCNN deep neural network, which is great at spotting faces in wild/real conditions.
   - Not just finds *where* faces are (bounding box), but finds facial landmarks (eyes, nose, mouth).

3. **Crop & Standardize Faces:**
   - Crops detected face area out of the frame.
   - Resizes face patch to 160x160 pixels (what our face embedding model expects).
   - Converts to RGB, normalizes pixel values, makes it ready for model input.

4. **FaceNet Embedding Extraction:**
   - The magic: pass cropped face into a pre-trained FaceNet network (InceptionResNetV1).
   - Output is a vector (512 real numbers) — this is the "face fingerprint." Every human’s face will be, probabilistically, very far from others in this space (if seen by this model).

5. **Similarity Search with FAISS:**
   - System keeps a big list (vector index) of all criminal/enrolled faces (previously embedded and stored in the database).
   - New embedding is compared to all stored embeddings. If a match is close enough (0.4+ cosine sim, for example): that’s our guy!
   - FAISS makes this fast, even if the database grows super huge.

6. **Incident Logging & Alerting:**
   - On match: logs the detected event (incident) in DB — including which camera, time, cropped image path, and how close the match was.
   - Instantly emails (or webhooks) anyone you configure, with all the details and (optionally) a face snapshot.
   - Uses cooldown logic, so you aren’t spammed with 100 emails for the same dude.

7. **Monitoring/Analysis:**
   - Live metrics via Prometheus: How many faces, frames, matches, alerts, FPS, etc.
   - Grafana dashboard (check /docs/) to see everything working/graphing in real time.

---

## 2. Actual AI & ML Used

- **MTCNN:** Deep multi-stage network for tough face detection — beats Haar cascade, works on heads at weird angles, glasses, etc.
- **FaceNet:** 'Golden standard' face embedding; open-source pre-trained, so no need for huge local dataset.
- **FAISS:** Vector index by Facebook — lets you instantly search thousands/millions of faces on modest hardware.
- **asyncio:** Everything async; processes many camera feeds and images at once without blocking.
- **SQLAlchemy + PostgreSQL:** All faces, embeddings, events, and camera settings are in a real relational DB.
- **FastAPI:** REST API for everything (enroll faces, live search, review results).

---

## 3. Why Do It This Way? (Why Not Something Simpler?)

- **Robustness:** MTCNN + FaceNet is still SOTA for unconstrained CCTV. You actually get alerts when it matters, not a thousand false alarms.
- **Scalability:** Once deployed, system can go from a few faces/cameras to several hundred without architectural changes.
- **Code for Everyone:** Every module is as simple and commented as possible. No magic variables, everything is named to explain itself.
- **Visual Logging:** Want to see why it thinks that’s a match? Snapshots and incident logs show everything step-by-step.

---

## 4. Project Layout (Where Code Actually Lives/Flows)

- **src/core/**
    - **detection_engine.py:** Runs video → frame → detect → embed → match → alert. Main ML loop, heavily commented.
    - **pipeline.py:** Defines single-frame analytics/embedding logic. Easy to debug and swap models here.
    - **detectors/, recognition/:** MTCNN and FaceNet logic.
    - **video_stream.py:** Handles video stream opening, re-connection, and next-frame read — only errors if camera totally off network.

- **src/database/**
    - **models.py:** DB tables described; each row/column explained. Faces, incidents, embeddings, etc.
    - **embedding_index.py:** How FAISS index is loaded, saved, and searched.
    - **services.py:** All the business logic needed for face enrollment, logging incidents, etc.

- **src/api/**
    - **main.py:** FastAPI app factory, routes, DI and middleware (metrics and error logging included).
    - **criminals.py, cameras.py, incidents.py:** Each API endpoint, with what each HTTP verb and argument means.
    - **auth.py:** JWT-based security, fine-tuned for two levels: admin (CRUD, enroll), operator (watch/search).
    - **schemas.py:** Pydantic models for pure type safety and automatic docs.

- **src/alerts/**
    - **service.py, notifiers.py:** Alert logic — easily swap in SMS, webhook, etc.
    - **cooldown.py:** Rate limiting for real-world practicality.

---

## 5. Example Flow (REALLY Step by Step for Reviewers)

1. **You enroll a criminal:**
   - POST /api/v1/criminals with JSON
   - POST /api/v1/criminals/ID/upload-photo with a jpeg

2. **His face embedding is stored:**
   - Image → aligns/crops → FaceNet → 512D vector in DB + FAISS

3. **He's seen again:**
   - Face detected in frame X 
   - Embedding vector computed
   - Closest match? Score > threshold? Yes? Alert/Log/Notify!

4. **You can live search:**
   - POST /api/v1/criminals/search-by-image with photo snippet
   - REST API returns closest matches — for review/testing

5. **Everything tracked:**
   - Each incident: camera, face count, timestamp, cropped image, similarity score; Prometheus metrics updated; logs available for postmortem.

---

## 6. How To Run or Debug (For Anyone)

### To enroll and test:
```
# create environment, install deps (see README)
python scripts/setup_database.py
python main.py api
python main.py start

# via REST (see Swagger at /docs)
# POST /api/v1/criminals { ... }
# POST /api/v1/criminals/{id}/upload-photo
# POST /api/v1/criminals/search-by-image
```

### To check what’s happening:
- View logs in ./logs/
- Check incidents table for alerts and matched faces
- Watch live metrics in Prometheus, graphs in Grafana (see docs/ for prebuilt dashboard)
- Health/metrics endpoints give live feedback for ops

---

## 7. FAQ (Every Reviewer Asks)

**Q: Can you retrain it?**
- Yes, but that's a project extension. Current pipeline is plug & play. You can swap FaceNet with another embeddor in src/core/recognition/.

**Q: What if the person’s photo quality is low?**
- More photos (angles, lighting) per person improve match score. The rest is optimized by MTCNN and FaceNet robustness.

**Q: What do you store?**
- Nothing you can’t audit: every detection, photo, embedding vector, alert, all have DB and filesystem artifacts.

**Q: Can I swap out MTCNN or FaceNet?**
- Yup. Just change the core modules (see comments!) and update config. The main loop is independent.

**Q: What hardware do I need?**
- Any decent CPU for a couple streams. GPU only needed for many cameras or max FPS.

---

## 8. For Future Improvement
- On-demand live streaming in the dashboard
- Add facial mask detection/post-processing (COVID/compliance/business rules)
- Human operator feedback loop (label incident in dashboard, help tune threshold)
- Edge deployment (Jetson, Coral, etc) for on-camera inferencing

---

*This documentation and in-code comments are designed so that: A new engineer, or a total outsider, could rebuild, demo, or extend this system from scratch in a weekend. Every rationale and choice is justified. Ask for richer AI/ML explanations anywhere—will auto-generate more detail on demand!*
