# Python 3.14 matches the AI package requirement. Debian Bookworm's glibc
# satisfies the MediaPipe Linux wheel (glibc 2.28+) and OpenCV wheels.
FROM python:3.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ATHLETELENS_POSE_MODEL=/app/ai_engine/models/pose_landmarker_lite.task \
    ATHLETELENS_ANNOTATED_VIDEO_DIR=/data/annotated_videos

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Keep only the shared installable AI package in the image, not videos or
# generated experimental outputs.
COPY ai_engine/pyproject.toml ai_engine/requirements.txt ./ai_engine/
COPY ai_engine/athletelens_ai ./ai_engine/athletelens_ai
COPY backend/requirements.txt ./backend/requirements.txt

# The task model is public, pinned by URL and SHA-256, and intentionally kept
# out of Git. Download it into the image so Railway does not need a model volume.
RUN mkdir -p /app/ai_engine/models \
    && python -c "from hashlib import sha256; from pathlib import Path; from urllib.request import urlopen; url='https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'; data=urlopen(url, timeout=90).read(); expected='59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a'; actual=sha256(data).hexdigest(); assert actual == expected, f'Pose model checksum mismatch: {actual}'; Path('/app/ai_engine/models/pose_landmarker_lite.task').write_bytes(data)"

WORKDIR /app/backend
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip check \
    && python -c "import cv2, mediapipe, numpy; from mediapipe.tasks.python import vision; print('cv2', cv2.__version__, 'mediapipe', mediapipe.__version__, 'numpy', numpy.__version__, 'PoseLandmarker', hasattr(vision, 'PoseLandmarker'))"

RUN groupadd --system --gid 10001 athletelens \
    && useradd --system --uid 10001 --gid athletelens --home-dir /app athletelens \
    && mkdir -p /data/models /data/annotated_videos \
    && chown -R athletelens:athletelens /app /data

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD-SHELL python -c 'import os; from urllib.request import urlopen; urlopen("http://127.0.0.1:" + os.getenv("PORT", "8000") + "/health", timeout=3).read()' || exit 1

CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
