  

  
  
  
  

set -euo pipefail

  
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

  
VENV_DIR="${VENV_DIR:-.venv}"

echo "Project root: ${PROJECT_ROOT}"
echo "Using virtual environment: ${PROJECT_ROOT}/${VENV_DIR}"

  
if [ ! -d "${VENV_DIR}" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "${VENV_DIR}"
else
  echo "Virtual environment already exists, skipping creation."
fi

  
if [ -f "${VENV_DIR}/bin/activate" ]; then
    
    
  source "${VENV_DIR}/bin/activate"
elif [ -f "${VENV_DIR}/Scripts/activate" ]; then
    
    
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

