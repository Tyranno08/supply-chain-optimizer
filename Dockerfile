# ============================================
# SUPPLY CHAIN OPTIMIZER — DOCKERFILE
# Multi-stage build for production deployment
# ============================================

# ---- Stage 1: Base Python environment ----
FROM python:3.10-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# ---- Stage 2: Install Python dependencies ----
FROM base AS dependencies

# Copy requirements first (Docker cache optimization)
COPY requirements.txt .

# Install all Python packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ---- Stage 3: Final production image ----
FROM dependencies AS production

# Copy source code
COPY src/         ./src/
COPY models/      ./models/
COPY data/processed/ ./data/processed/
COPY .env         ./.env

# Create logs directory
RUN mkdir -p logs

# Expose API port
EXPOSE 8000

# Health check — Docker will restart container if this fails
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI server
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
