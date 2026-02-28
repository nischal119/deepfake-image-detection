"""
Robustness tests for deepfake detection models.

Creates degraded video variants (bitrate, resolution, JPEG compression, Gaussian blur),
runs inference on each, and reports metric degradation (delta AUC, delta accuracy).
Outputs matplotlib figures and a CSV summary.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torchvision import models, transforms
from tqdm.auto import tqdm

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]

# Default frame transform (ImageNet normalization)
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _read_middle_frame(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = frame_count // 2 if frame_count > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def default_frame_inference_fn(
    video_paths: Sequence[Path],
    model_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Run frame-level inference (middle frame per video) using ResNet-50 checkpoint.

    Returns array of shape (N,) with fake probabilities for each video.
    """
    if model_path is None:
        model_path = PROJECT_ROOT / "models" / "frame_resnet50.pth"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)

    # Build model matching ResNet50FrameClassifier (backbone + classifier)
    backbone = models.resnet50(weights=None)
    in_features = backbone.fc.in_features
    backbone.fc = torch.nn.Identity()

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.classifier = torch.nn.Linear(in_features, 2)

        def forward(self, x):
            return self.classifier(self.backbone(x))

    model = _Model()
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    probs = []
    with torch.no_grad():
        for p in video_paths:
            frame = _read_middle_frame(p)
            x = DEFAULT_TRANSFORM(frame).unsqueeze(0).to(device)
            logits = model(x)
            prob_fake = torch.softmax(logits, dim=1)[0, 1].item()
            probs.append(prob_fake)
    return np.array(probs)


# --- Degradation functions ---


def degrade_bitrate(input_path: Path, output_path: Path, bitrate_kbps: int = 500) -> Path:
    """Re-encode with lower bitrate via ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-b:v", f"{bitrate_kbps}k", "-bufsize", f"{bitrate_kbps * 2}k",
            "-c:a", "copy",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def degrade_resolution(
    input_path: Path,
    output_path: Path,
    width: int = 640,
    height: int = 360,
) -> Path:
    """Downscale to target resolution (e.g., 360p)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-vf", f"scale={width}:{height}",
            "-c:a", "copy",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def degrade_jpeg(
    input_path: Path,
    output_path: Path,
    quality: int = 50,
) -> Path:
    """Re-encode frames with JPEG compression (lower quality = more compression)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        decoded = cv2.imdecode(jpeg_buf, cv2.IMREAD_COLOR)
        writer.write(decoded)
    cap.release()
    writer.release()
    return output_path


def degrade_blur(input_path: Path, output_path: Path, kernel_size: int = 15) -> Path:
    """Apply Gaussian blur to each frame."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        blurred = cv2.GaussianBlur(frame, (k, k), 0)
        writer.write(blurred)
    cap.release()
    writer.release()
    return output_path


DEGRADATION_FUNCS = {
    "bitrate_500k": lambda src, dst: degrade_bitrate(src, dst, bitrate_kbps=500),
    "resolution_360p": lambda src, dst: degrade_resolution(src, dst, width=640, height=360),
    "jpeg_quality_50": lambda src, dst: degrade_jpeg(src, dst, quality=50),
    "gaussian_blur_15": lambda src, dst: degrade_blur(src, dst, kernel_size=15),
}


def create_degraded_variants(
    video_paths: Sequence[Path],
    output_dir: Path,
    degradations: Optional[Dict[str, Callable]] = None,
) -> Dict[str, List[Path]]:
    """
    Create degraded copies for each video.

    Returns dict: {degradation_name: [path1, path2, ...]}
    """
    output_dir = Path(output_dir)
    degradations = degradations or DEGRADATION_FUNCS
    variants: Dict[str, List[Path]] = {name: [] for name in degradations}

    for i, src in enumerate(tqdm(video_paths, desc="Creating degraded variants")):
        stem = src.stem
        for name, fn in degradations.items():
            dst = output_dir / name / f"{stem}_{i}.mp4"
            try:
                fn(Path(src), dst)
                variants[name].append(dst)
            except Exception as e:
                tqdm.write(f"Warning: {name} failed for {src}: {e}")
    return variants


def run_robustness_tests(
    videos: List[Tuple[Path, int]],
    inference_fn: Optional[Callable[[Sequence[Path]], np.ndarray]] = None,
    output_dir: Path = PROJECT_ROOT / "results" / "robustness",
    degradations: Optional[Dict[str, Callable]] = None,
) -> pd.DataFrame:
    """
    Run robustness tests: create variants, run inference, compute metrics.

    Parameters
    ----------
    videos : list of (path, label)
    inference_fn : callable(paths) -> probs array. If None, uses default frame model.
    output_dir : where to write degraded videos, graphs, CSV
    degradations : dict of name -> (src, dst) -> Path. Default: bitrate, resolution, jpeg, blur.

    Returns
    -------
    summary_df : DataFrame with columns [degradation, auc, accuracy, delta_auc, delta_accuracy]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [p for p, _ in videos]
    labels = np.array([lbl for _, lbl in videos], dtype=int)

    if inference_fn is None:
        def _inference(ps):
            return default_frame_inference_fn(ps)
        inference_fn = _inference

    # Baseline on original videos
    tqdm.write("Running baseline inference...")
    baseline_probs = inference_fn(paths)
    baseline_preds = (baseline_probs >= 0.5).astype(int)
    try:
        baseline_auc = roc_auc_score(labels, baseline_probs)
    except ValueError:
        baseline_auc = float("nan")
    baseline_acc = accuracy_score(labels, baseline_preds)

    rows = [
        {
            "degradation": "baseline",
            "auc": baseline_auc,
            "accuracy": baseline_acc,
            "delta_auc": 0.0,
            "delta_accuracy": 0.0,
        }
    ]

    # Create degraded variants and evaluate
    with tempfile.TemporaryDirectory(prefix="robustness_", dir=output_dir) as tmpdir:
        tmpdir = Path(tmpdir)
        variants = create_degraded_variants(paths, tmpdir, degradations)

        for name, degraded_paths in variants.items():
            if len(degraded_paths) != len(paths):
                tqdm.write(f"Skipping {name}: only {len(degraded_paths)}/{len(paths)} succeeded")
                continue
            tqdm.write(f"Evaluating {name}...")
            probs = inference_fn(degraded_paths)
            preds = (probs >= 0.5).astype(int)
            try:
                auc = roc_auc_score(labels, probs)
            except ValueError:
                auc = float("nan")
            acc = accuracy_score(labels, preds)
            delta_auc = (auc - baseline_auc) if not np.isnan(baseline_auc) else float("nan")
            delta_acc = acc - baseline_acc
            rows.append({
                "degradation": name,
                "auc": auc,
                "accuracy": acc,
                "delta_auc": delta_auc,
                "delta_accuracy": delta_acc,
            })

    summary_df = pd.DataFrame(rows)
    csv_path = output_dir / "robustness_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = range(len(summary_df))
    labels_plot = summary_df["degradation"].tolist()
    axes[0].bar(x, summary_df["auc"], color="steelblue", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels_plot, rotation=45, ha="right")
    axes[0].set_ylabel("AUC")
    axes[0].set_title("AUC per Degradation")
    axes[0].axhline(y=baseline_auc, color="gray", linestyle="--", alpha=0.7)

    axes[1].bar(x, summary_df["accuracy"], color="coral", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels_plot, rotation=45, ha="right")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy per Degradation")
    axes[1].axhline(y=baseline_acc, color="gray", linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig_path = output_dir / "robustness_metrics.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {fig_path}")

    # Delta plot
    fig2, ax = plt.subplots(figsize=(8, 4))
    width = 0.35
    x_pos = np.arange(len(summary_df) - 1)  # exclude baseline
    ax.bar(x_pos - width / 2, summary_df.loc[1:, "delta_auc"], width, label="Delta AUC")
    ax.bar(x_pos + width / 2, summary_df.loc[1:, "delta_accuracy"], width, label="Delta Accuracy")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(summary_df.loc[1:, "degradation"], rotation=45, ha="right")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend()
    ax.set_title("Metric Degradation vs Baseline")
    plt.tight_layout()
    delta_path = output_dir / "robustness_delta.png"
    fig2.savefig(delta_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {delta_path}")

    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run robustness tests on deepfake detector.")
    parser.add_argument(
        "--videos-dir",
        type=str,
        default=None,
        help="Directory with videos (default: <repo>/deepfake-videos). Uses label inference from paths.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=50,
        help="Max videos to test (for speed).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "robustness"),
    )
    return parser.parse_args()


def _collect_labeled_videos(root: Path, max_videos: int) -> List[Tuple[Path, int]]:
    from src.train.train_frame_model import collect_labeled_videos
    labeled, _ = collect_labeled_videos(root)
    if max_videos and len(labeled) > max_videos:
        labeled = labeled[:max_videos]
    return labeled


def main() -> None:
    args = parse_args()
    root = Path(args.videos_dir).expanduser().resolve() if args.videos_dir else PROJECT_ROOT.parent / "deepfake-videos"
    if not root.exists():
        raise SystemExit(f"Videos directory does not exist: {root}")

    videos = _collect_labeled_videos(root, args.max_videos)
    print(f"Testing on {len(videos)} videos from {root}")

    run_robustness_tests(
        videos=videos,
        inference_fn=None,
        output_dir=Path(args.output_dir).expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
