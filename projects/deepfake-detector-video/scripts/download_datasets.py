from __future__ import annotations

"""
Helpers for downloading and organizing external video deepfake datasets
for the video module:

- FaceForensics++
- Facebook Deepfake Detection Challenge (DFDC)

This file is intentionally light on implementation details for the actual
downloads because both datasets typically require manual authentication
or acceptance of terms. Instead, it:

- Creates the expected directory structure under `data/raw_videos/`.
- Documents where to place downloaded archives.
- Provides placeholder functions you can extend with your own API logic.
- Exposes `verify_dataset_presence()` to sanity-check the layout.
"""

import itertools
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_VIDEOS_DIR = PROJECT_ROOT / "data" / "raw_videos"

FACEFORENSICSPP_DIR = RAW_VIDEOS_DIR / "faceforensicspp"
DFDC_DIR = RAW_VIDEOS_DIR / "dfdc"


def _ensure_dir(path: Path) -> None:
    """Create directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def create_dataset_dirs() -> None:
    """
    Ensure that the canonical raw-video folders exist for external datasets.

    - data/raw_videos/faceforensicspp/
    - data/raw_videos/dfdc/
    """
    _ensure_dir(FACEFORENSICSPP_DIR)
    _ensure_dir(DFDC_DIR)


def download_faceforensicspp() -> None:
    """
    Placeholder for FaceForensics++ download logic.

    FaceForensics++ generally requires:
    - Registration / approval on the official site.
    - Manual download of archives (or use of provided scripts with credentials).

    Recommended workflow:
    1. Request access and download the desired FaceForensics++ splits.
    2. Place the downloaded zip/tar files under:

         data/raw_videos/faceforensicspp/

       For example:
         data/raw_videos/faceforensicspp/original_videos.zip
         data/raw_videos/faceforensicspp/manipulated_videos.zip

    3. Extract the archives *in place* so that the final structure looks like:

         data/raw_videos/faceforensicspp/
           original/...
           manipulated/...

    If you have programmatic credentials or a private mirror, replace the body
    of this function with your actual download code (e.g., requests, gdown,
    cloud SDKs, etc.).
    """
    create_dataset_dirs()
    print(f"[FaceForensics++] Expected root: {FACEFORENSICSPP_DIR}")
    print(
        "Please manually download FaceForensics++ archives from the official site\n"
        "and place them under this folder, then extract them in place.\n"
        "Once extracted, videos should live somewhere under this directory."
    )


def download_dfdc() -> None:
    """
    Placeholder for DFDC (Deepfake Detection Challenge) download logic.

    The original DFDC dataset is quite large and often distributed via:
    - Official Kaggle competition page (multiple zip parts), or
    - Other research mirrors that may require authentication.

    Recommended workflow:
    1. Download DFDC train/test archives from Kaggle or your approved source.
    2. Place the zip files under:

         data/raw_videos/dfdc/

       For example:
         data/raw_videos/dfdc/train_part_00.zip
         data/raw_videos/dfdc/train_part_01.zip
         ...

    3. Extract all archives *in place* so that the final structure looks like:

         data/raw_videos/dfdc/
           train/...
           test/...
           metadata.json / metadata.csv (if provided)

    If you have programmatic access (e.g., Kaggle API token configured),
    you can replace the body of this function with code that uses the
    Kaggle API or `kagglehub` to automate the download.
    """
    create_dataset_dirs()
    print(f"[DFDC] Expected root: {DFDC_DIR}")
    print(
        "Please manually download DFDC archives from Kaggle or your approved source\n"
        "and place them under this folder, then extract them in place.\n"
        "Once extracted, videos should live somewhere under this directory."
    )


def _iter_video_files(root: Path, exts: Iterable[str]) -> Iterable[Path]:
    """Yield all video files under root with given extensions."""
    lowered_exts = {e.lower() for e in exts}
    if not root.exists():
        return []
    return (
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in lowered_exts
    )


def verify_dataset_presence() -> None:
    """
    Check whether the expected dataset directories and some representative files exist.

    This does *not* exhaustively validate checksums or completeness, but it gives
    a quick sanity check that:
    - The base folders exist.
    - There is at least one video file (mp4/avi/mov/mkv) in each dataset.

    Prints a summary of what is present and what appears to be missing.
    """
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
            print("Status: MISSING (directory does not exist)")
            missing_any = True
            continue

        videos = list(_iter_video_files(root, video_exts))
        if not videos:
            print("Status: DIRECTORY PRESENT, BUT NO VIDEO FILES FOUND")
            print(
                "Hint: Ensure you've extracted the downloaded archives so that\n"
                "      video files (e.g., .mp4) live under this directory."
            )
            missing_any = True
        else:
            print(f"Status: OK ({len(videos)} video file(s) detected)")
            print("Sample files:")
            for p in itertools.islice(videos, 5):
                print("  -", p.relative_to(PROJECT_ROOT))

    if not missing_any:
        print("\nAll expected datasets appear to be present (at least one video each).")
    else:
        print(
            "\nSome datasets or files appear to be missing. Please review the messages above\n"
            "and make sure you've downloaded and extracted the archives into the expected folders."
        )


if __name__ == "__main__":
    create_dataset_dirs()
    verify_dataset_presence()

