from __future__ import annotations

"""
Helpers for creating the required dataset directories for:
- FaceForensics++
- Facebook Deepfake Detection Challenge (DFDC)
"""

import itertools
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_VIDEOS_DIR = PROJECT_ROOT / "data" / "raw_videos"

FACEFORENSICSPP_DIR = RAW_VIDEOS_DIR / "faceforensicspp"
DFDC_DIR = RAW_VIDEOS_DIR / "dfdc"


def create_dataset_dirs() -> None:
     
    FACEFORENSICSPP_DIR.mkdir(parents=True, exist_ok=True)
    DFDC_DIR.mkdir(parents=True, exist_ok=True)


def _iter_video_files(root: Path, exts: Iterable[str]) -> Iterable[Path]:
     
    lowered_exts = {e.lower() for e in exts}
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in lowered_exts)


def verify_dataset_presence() -> None:
     
    create_dataset_dirs()

    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    datasets = [
        ("FaceForensics++", FACEFORENSICSPP_DIR),
        ("DFDC", DFDC_DIR),
    ]

    missing_any = False

    for name, root in datasets:
        print(f"\n=== {name} ===")
        print(f"Expected root: {root}")

        if not root.exists():
            print("Status: MISSING")
            missing_any = True
            continue

        videos = list(_iter_video_files(root, video_exts))
        if not videos:
            print("Status: EMPTY")
            missing_any = True
        else:
            print(f"Status: OK ({len(videos)} videos found)")
            for p in itertools.islice(videos, 5):
                print("  -", p.relative_to(PROJECT_ROOT))

    if not missing_any:
        print("\nAll datasets are present.")
    else:
        print("\nSome datasets are missing. Please download and extract them manually.")


if __name__ == "__main__":
    create_dataset_dirs()
    verify_dataset_presence()

