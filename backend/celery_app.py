"""Celery application entry point."""

import os

import sys
from celery import Celery

backend_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPATH"] = backend_dir + os.pathsep + os.environ.get("PYTHONPATH", "")

if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

if "" not in sys.path:
    sys.path.insert(0, "")

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
)
