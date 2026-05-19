# syntax=docker/dockerfile:1.7

# ── Stage 1: Build the Vite frontend ────────────────────────────────────────
FROM node:22-alpine AS frontend
WORKDIR /web

COPY package.json package-lock.json* ./
RUN npm ci

COPY tsconfig*.json vite.config.ts index.html ./
COPY src ./src

# Empty base URL → frontend uses relative /api/... served by FastAPI.
ENV VITE_OCR_API_URL=""
RUN npm run build


# ── Stage 2: Python backend + frontend static files ─────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STATIC_DIR=/app/static \
    PORT=8080

# Native deps for opencv-python-headless, pymupdf, easyocr (torch).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 libgl1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch wheel first (smaller, avoids CUDA).
COPY python/requirements.txt /tmp/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.6.0+cpu torchvision==0.21.0+cpu \
    && pip install -r /tmp/requirements.txt

# Backend source
COPY python/ /app/

# Frontend dist → /app/static (served by FastAPI when STATIC_DIR is set)
COPY --from=frontend /web/dist /app/static

EXPOSE 8080

# Cloud Run injects $PORT (default 8080). Bind 0.0.0.0.
CMD ["sh", "-c", "exec uvicorn api.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
