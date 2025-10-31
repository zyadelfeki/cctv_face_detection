# CCTV Face Detection - Operational Runbook

## Setup
- Configure environment: `.env` and `config/config.yaml`.
- Run migrations and initialize storage: `python scripts/setup_database.py`.
- Seed demo data (optional): `python scripts/seed_demo_data.py`.

## Enrollment Workflow
1. Create a criminal record: `POST /api/v1/criminals` (admin).
2. Upload photo(s): `POST /api/v1/criminals/{id}/upload-photo` (admin).
3. Search-by-image to validate: `POST /api/v1/criminals/search-by-image`.

## Running
- API server: `python main.py api` (or via Docker Compose).
- Detection engine: `python main.py start` (requires configured RTSP sources).

## Alerts
- Configure SMTP/webhook in `config/config.yaml` under `alerting`.
- Cooldown: 300s per criminal per camera.

## Maintenance
- Rebuild FAISS index: `python scripts/rebuild_index.py`.
- Backup DB: dump PostgreSQL; backup `./data` for FAISS and snapshots.

## Troubleshooting
- Check logs in `./logs`.
- Health: `GET /health`.
- Metrics: `GET /metrics` (Prometheus format).
