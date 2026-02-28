"""Flask and Celery configuration."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    DATABASE_URL = os.environ.get(
        "DATABASE_URL",
        "postgresql://localhost/deepfake_video",
    )
    UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", BASE_DIR / "uploads"))
    MAX_VIDEO_SIZE_MB = int(os.environ.get("MAX_VIDEO_SIZE_MB", 100))
    ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
    CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    RATE_LIMIT = os.environ.get("RATE_LIMIT", "300 per minute")


def init_upload_dir(app_config: dict) -> None:
    up = app_config.get("UPLOAD_DIR", BASE_DIR / "uploads")
    Path(up).mkdir(parents=True, exist_ok=True)
