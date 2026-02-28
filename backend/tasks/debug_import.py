import sys
import os
from pathlib import Path

file_path = Path(__file__).resolve()
backend_root = file_path.parents[1]
app_path = backend_root / "app"
init_file = app_path / "__init__.py"

print(f"File: {file_path}")
print(f"Backend Root: {backend_root}")
print(f"App Path: {app_path}")
print(f"Init Exists: {init_file.exists()}")

sys.path.insert(0, str(backend_root))
try:
    import app
    print("Import app: SUCCESS")
    print(f"App file: {app.__file__}")
except Exception as e:
    print(f"Import app: FAILED - {e}")
