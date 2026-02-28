"""
Video inference: extract frames, run optimized model, return JSON with scores.

Supports TorchScript (.pt), ONNX (.onnx), or native PyTorch (.pth) checkpoints.
Includes gradient-based saliency heatmap generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from torchvision import transforms

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DEPLOY_DIR = PROJECT_ROOT / "deploy"

# Sensitivity: lower temperature = more decisive/sensitive predictions
SENSITIVITY_TEMPERATURE = 0.5

# Default preprocess for ResNet-based frame model
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------------------------------------------------------------------
# Saliency / Heatmap Generation
# ---------------------------------------------------------------------------

def generate_saliency_map(
    model: Union[torch.jit.ScriptModule, torch.nn.Module],
    frame: np.ndarray,
    transform: transforms.Compose,
    device: torch.device,
) -> Optional[np.ndarray]:
    """Generate gradient-based saliency heatmap overlaid on the original frame.

    Works with both native PyTorch and TorchScript models.
    Returns an RGB overlay image (np.ndarray) or None on failure.
    """
    try:
        tensor = transform(frame).unsqueeze(0).to(device)
        tensor = tensor.clone().detach().requires_grad_(True)

        logits = model(tensor)
        fake_prob = torch.softmax(logits / SENSITIVITY_TEMPERATURE, dim=1)[0, 1]
        fake_prob.backward()

        if tensor.grad is None:
            return None

        # Max absolute gradient across RGB channels → pixel importance
        saliency = tensor.grad.abs().squeeze().max(dim=0)[0].cpu().numpy()
        saliency = cv2.GaussianBlur(saliency, (21, 21), 0)

        s_min, s_max = saliency.min(), saliency.max()
        if s_max - s_min < 1e-8:
            return None
        saliency = (saliency - s_min) / (s_max - s_min)

        h, w = frame.shape[:2]
        saliency_u8 = cv2.resize((saliency * 255).astype(np.uint8), (w, h))

        heatmap_bgr = cv2.applyColorMap(saliency_u8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(frame, 0.55, heatmap_rgb, 0.45, 0)
        return overlay
    except Exception as e:
        print(f"Warning: Saliency map generation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Frame Extraction
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: Path,
    num_frames: int = 16,
    strategy: str = "uniform",
) -> List[np.ndarray]:
    """Extract frames from video.

    strategy: "uniform" (spread across video) or "middle" (single middle frame).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if frame_count <= 0:
        raise RuntimeError(f"No frames in video: {video_path}")

    if strategy == "middle" or num_frames == 1:
        frame_indices = [max(0, frame_count // 2)]
    else:
        lin = np.linspace(0, frame_count - 1, num_frames)
        frame_indices = np.round(lin).astype(int).tolist()

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_torchscript_model(path: Path, device: torch.device) -> torch.jit.ScriptModule:
    model = torch.jit.load(str(path), map_location=device)
    model.eval()
    return model


def load_onnx_model(path: Path) -> "onnx.InferenceSession":
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("Install onnxruntime: pip install onnxruntime")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def load_native_model(path: Path, device: torch.device) -> torch.nn.Module:
    from src.train.train_frame_model import ResNet50FrameClassifier
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model = ResNet50FrameClassifier(num_classes=2, pretrained=False)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference helpers (with temperature scaling)
# ---------------------------------------------------------------------------

def infer_torchscript(
    model: torch.jit.ScriptModule,
    frames: List[np.ndarray],
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int = 8,
) -> List[float]:
    probs = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        tensors = torch.stack([transform(f) for f in batch]).to(device)
        with torch.no_grad():
            logits = model(tensors)
            p = torch.softmax(logits / SENSITIVITY_TEMPERATURE, dim=1)[:, 1].cpu().numpy()
        probs.extend(p.tolist())
    return probs


def infer_native(
    model: torch.nn.Module,
    frames: List[np.ndarray],
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int = 8,
) -> List[float]:
    probs = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        tensors = torch.stack([transform(f) for f in batch]).to(device)
        with torch.no_grad():
            logits = model(tensors)
            p = torch.softmax(logits / SENSITIVITY_TEMPERATURE, dim=1)[:, 1].cpu().numpy()
        probs.extend(p.tolist())
    return probs


def infer_onnx(
    sess: "onnx.InferenceSession",
    frames: List[np.ndarray],
    transform: transforms.Compose,
    batch_size: int = 8,
) -> List[float]:
    probs = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        tensors = torch.stack([transform(f) for f in batch]).numpy()
        out = sess.run(None, {"input": tensors})[0]
        p = torch.softmax(torch.from_numpy(out) / SENSITIVITY_TEMPERATURE, dim=1)[:, 1].numpy()
        probs.extend(p.tolist())
    return probs


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def predict_video(
    video_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    num_frames: int = 16,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    heatmap_dir: Optional[Union[str, Path]] = None,
) -> dict:
    """Run inference on a video; return dict with frame scores, video score, and heatmap paths.

    Args:
        heatmap_dir: If provided, gradient-saliency heatmaps are saved here as PNGs.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path is None:
        candidates = [
            DEPLOY_DIR / "frame_resnet50.pt",
            DEPLOY_DIR / "frame_resnet50_fp16.pt",
            MODELS_DIR / "frame_resnet50.pth",
        ]
        for p in candidates:
            if p.exists():
                model_path = p
                break
        else:
            raise FileNotFoundError(
                "No model found. Export first: python -m src.deploy.optimize_export"
            )
    model_path = Path(model_path)

    frames = extract_frames(video_path, num_frames=num_frames, strategy="uniform")
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")

    suffix = model_path.suffix.lower()
    model = None
    if suffix == ".pt":
        model = load_torchscript_model(model_path, device)
        frame_scores = infer_torchscript(model, frames, DEFAULT_TRANSFORM, device, batch_size)
    elif suffix == ".onnx":
        sess = load_onnx_model(model_path)
        frame_scores = infer_onnx(sess, frames, DEFAULT_TRANSFORM, batch_size)
    else:
        model = load_native_model(model_path, device)
        frame_scores = infer_native(model, frames, DEFAULT_TRANSFORM, device, batch_size)

    # Generate saliency heatmaps
    heatmap_paths: Dict[int, str] = {}
    if heatmap_dir and model is not None:
        hm_dir = Path(heatmap_dir)
        hm_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            overlay = generate_saliency_map(model, frame, DEFAULT_TRANSFORM, device)
            if overlay is not None:
                out_path = hm_dir / f"frame_{i}.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                heatmap_paths[i] = str(out_path)

    video_score = float(np.mean(frame_scores))
    # Lowered threshold for higher sensitivity
    prediction = "fake" if video_score >= 0.35 else "real"

    return {
        "video_path": str(video_path),
        "num_frames": len(frame_scores),
        "frame_scores": [round(s, 4) for s in frame_scores],
        "video_score": round(video_score, 4),
        "prediction": prediction,
        "model_path": str(model_path),
        "heatmap_paths": heatmap_paths,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deepfake inference on a video.")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--heatmap-dir", type=str, default=None, help="Directory to save heatmaps")
    parser.add_argument("--output", type=str, default=None, help="Write JSON to file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = predict_video(
            args.video,
            model_path=args.model,
            num_frames=args.num_frames,
            batch_size=args.batch_size,
            heatmap_dir=args.heatmap_dir,
        )
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

    out = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
