#!/usr/bin/env bash

# Setup script for the video deepfake detector environment.
# - Creates (or reuses) a Python virtual environment
# - Installs dependencies from requirements.txt
# - Checks for ffmpeg and prints installation hints if missing

set -euo pipefail

# Resolve project root as the parent of this scripts directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Virtual environment directory (override by exporting VENV_DIR if desired)
VENV_DIR="${VENV_DIR:-.venv}"

echo "Project root: ${PROJECT_ROOT}"
echo "Using virtual environment: ${PROJECT_ROOT}/${VENV_DIR}"

# 1) Create Python virtualenv (idempotent)
if [ ! -d "${VENV_DIR}" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "${VENV_DIR}"
else
  echo "Virtual environment already exists, skipping creation."
fi

# 2) Activate virtualenv and install requirements
if [ -f "${VENV_DIR}/bin/activate" ]; then
  # Unix-like (macOS/Linux)
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
elif [ -f "${VENV_DIR}/Scripts/activate" ]; then
  # Windows (Git Bash, WSL)
  # shellcheck disable=SC1091
  source "${VENV_DIR}/Scripts/activate"
else
  echo "Could not find activate script in ${VENV_DIR}. Aborting." >&2
  exit 1
fi

if [ ! -f "requirements.txt" ]; then
  echo "requirements.txt not found in ${PROJECT_ROOT}. Aborting." >&2
  exit 1
fi

echo "Installing Python dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 3) Check for ffmpeg and print instructions if missing
if command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is installed: $(command -v ffmpeg)"
else
  echo "ffmpeg is NOT installed on this system."
  echo "Please install ffmpeg before running video processing scripts. Examples:"
  echo "  macOS (Homebrew):  brew install ffmpeg"
  echo "  Ubuntu/Debian:     sudo apt-get update && sudo apt-get install ffmpeg"
  echo "  Fedora:            sudo dnf install ffmpeg"
  echo "  Windows (choco):   choco install ffmpeg"
fi

echo "Environment setup complete."

