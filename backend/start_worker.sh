#!/bin/bash
cd "$(dirname "$0")"

BACKEND_DIR=$(pwd)
VIDEO_PROJECT_DIR="$(dirname "$BACKEND_DIR")/projects/deepfake-detector-video"
export PYTHONPATH="$BACKEND_DIR:$VIDEO_PROJECT_DIR:$PYTHONPATH"

echo "Starting Celery worker for DeepFake detection..."
echo "PYTHONPATH is set to: $PYTHONPATH"
python -c "import numpy; print(f'Numpy version: {numpy.__version__}')"

# Use a single-process worker to avoid macOS fork/signal issues and prevent multiple heavy jobs running concurrently.
celery -A celery_app worker --loglevel=info -P solo --concurrency=1 --without-heartbeat --without-gossip --without-mingle
