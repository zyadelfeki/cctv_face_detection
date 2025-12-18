# =============================================================================
# CCTV Face Detection System - Production Dockerfile
# Multi-stage build for security and minimal size
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Compile dependencies and prepare application
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Labels
LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.authors="zyadelfeki" \
      org.opencontainers.image.url="https://github.com/zyadelfeki/cctv_face_detection" \
      org.opencontainers.image.source="https://github.com/zyadelfeki/cctv_face_detection" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.title="CCTV Face Detection" \
      org.opencontainers.image.description="Secure CCTV facial recognition system"

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    libopencv-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install dependencies
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY pyproject.toml .

# -----------------------------------------------------------------------------
# Stage 2: Security Scanner
# -----------------------------------------------------------------------------
FROM builder AS security-scan

# Install security scanning tools
RUN pip install --no-cache-dir bandit safety

# Run security scans
RUN bandit -r src/ -f json -o /tmp/bandit-report.json || true
RUN safety check --json > /tmp/safety-report.json || true

# -----------------------------------------------------------------------------
# Stage 3: Runtime - Minimal production image
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -g 10000 appuser && \
    useradd -u 10000 -g appuser -s /sbin/nologin -c "Application User" appuser

# Create application directories
RUN mkdir -p /app /app/logs /app/cache /tmp/app && \
    chown -R appuser:appuser /app /tmp/app

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# Copy application code
COPY --from=builder --chown=appuser:appuser /build/src /app/src
COPY --from=builder --chown=appuser:appuser /build/alembic /app/alembic
COPY --from=builder --chown=appuser:appuser /build/alembic.ini /app/
COPY --from=builder --chown=appuser:appuser /build/pyproject.toml /app/

# Set working directory
WORKDIR /app

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    ENV=production \
    LOG_LEVEL=INFO

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# -----------------------------------------------------------------------------
# Stage 4: Distroless (Optional - ultra-minimal)
# -----------------------------------------------------------------------------
FROM gcr.io/distroless/python3-debian11 AS distroless

# Copy from runtime
COPY --from=runtime --chown=10000:10000 /opt/venv /opt/venv
COPY --from=runtime --chown=10000:10000 /app /app

# Set environment
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Run as non-root
USER 10000

# Expose port
EXPOSE 8000

# Entry point
ENTRYPOINT ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
