#!/bin/bash
# Script to launch the DeepFake Detector Celery worker correctly

# Navigate to the backend directory
cd "$(dirname "$0")"

# Set PYTHONPATH to include the current directory (backend/)
# This ensures that child processes created by Celery can always find the 'app' module.
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Starting Celery worker for DeepFake detection..."
echo "PYTHONPATH is set to: $PYTHONPATH"
python -c "import numpy; print(f'Numpy version: {numpy.__version__}')"

# Run the worker with solo pool for better stability on macOS
celery -A celery_app worker --loglevel=info -P solo
