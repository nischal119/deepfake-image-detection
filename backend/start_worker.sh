#!/bin/bash
# Script to launch the DeepFake Detector Celery worker correctly

# Navigate to the backend directory
cd "$(dirname "$0")"

# Set PYTHONPATH to include the backend/ and the video project
# This ensures Celery can find both 'app' and 'src.deploy.infer_video'
BACKEND_DIR=$(pwd)
VIDEO_PROJECT_DIR="$(dirname "$BACKEND_DIR")/projects/deepfake-detector-video"
export PYTHONPATH="$BACKEND_DIR:$VIDEO_PROJECT_DIR:$PYTHONPATH"

echo "Starting Celery worker for DeepFake detection..."
echo "PYTHONPATH is set to: $PYTHONPATH"
python -c "import numpy; print(f'Numpy version: {numpy.__version__}')"

# Run the worker with solo pool for better stability on macOS
# Disable celery heartbeat/gossip/mingle to avoid issues in some environments
# (e.g. "Load averages are unobtainable"). Also force concurrency=1 to avoid
# multiple heavy video jobs running at once if multiple workers are launched.
celery -A celery_app worker --loglevel=info -P solo --concurrency=1 --without-heartbeat --without-gossip --without-mingle
