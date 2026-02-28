"""Celery application configuration."""

import os

from celery import Celery

celery = Celery(
    "video_processor",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
    include=["tasks.process_video"],
)
celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_routes={"tasks.process_video.process_video": {"queue": "video"}},
)
