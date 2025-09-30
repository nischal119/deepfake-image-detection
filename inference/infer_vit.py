#!/usr/bin/env python3
import argparse
import base64
import io
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


def load_model(checkpoint_dir: Path):
    # Prefer a local fine-tuned checkpoint if provided; otherwise fall back to hub id
    if checkpoint_dir and checkpoint_dir.exists():
        model = ViTForImageClassification.from_pretrained(str(checkpoint_dir))
        # Try to reconstruct processor from config; if missing, default to a common ViT
        try:
            processor = ViTImageProcessor.from_pretrained(str(checkpoint_dir))
        except Exception:
            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    else:
        model_id = os.environ.get(
            "VIT_MODEL_ID", "dima806/deepfake_vs_real_image_detection"
        )
        processor = ViTImageProcessor.from_pretrained(model_id)
        model = ViTForImageClassification.from_pretrained(model_id)
    model.eval()
    return model, processor


def predict(
    model, processor, image_path: Path, temperature: float = 1.0, explain: bool = False
):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # temperature scaling to avoid over-confident 0/1
    model.eval()
    inputs = {k: v.requires_grad_(explain) for k, v in inputs.items()}
    outputs = model(**inputs)
    logits = outputs.logits[0] / max(1e-6, float(temperature))
    probs = torch.softmax(logits, dim=-1)

    # Assume label mapping: index 0 -> Real, 1 -> Fake as in training script
    prob_fake = float(probs[1].item())
    prob_real = float(probs[0].item())

    if prob_fake > 0.7:
        verdict = "likely_fake"
    elif prob_fake < 0.3:
        verdict = "likely_real"
    else:
        verdict = "inconclusive"

    result = {
        "score": prob_fake,
        "verdict": verdict,
        "probs": {"real": prob_real, "fake": prob_fake},
    }

    if explain:
        try:
            # Simple saliency: grad of fake logit w.r.t input pixels
            model.zero_grad(set_to_none=True)
            one_hot = torch.zeros_like(logits)
            one_hot[1] = 1.0  # fake class
            (logits * one_hot).sum().backward()
            grad = inputs["pixel_values"].grad  # (1,3,H,W)
            g = grad.abs().sum(dim=1)[0]  # (H,W)
            g = (g - g.min()) / (g.max() - g.min() + 1e-8)
            g_np = g.detach().cpu().numpy()
            # Colorize heatmap (red) and overlay
            heat = np.zeros((g_np.shape[0], g_np.shape[1], 3), dtype=np.float32)
            heat[..., 0] = g_np  # red channel
            heat_img = (heat * 255).astype(np.uint8)
            heat_pil = Image.fromarray(heat_img).resize(image.size)
            overlay = Image.blend(
                image.convert("RGBA"), heat_pil.convert("RGBA"), alpha=0.35
            )
            buf = io.BytesIO()
            overlay.save(buf, format="PNG")
            heatmap_dataurl = "data:image/png;base64," + base64.b64encode(
                buf.getvalue()
            ).decode("ascii")
            result["heatmap"] = heatmap_dataurl
        except Exception:
            pass

        try:
            # Lightweight artifact metrics
            im_gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            # High-frequency energy ratio via FFT
            f = np.fft.fftshift(np.fft.fft2(im_gray))
            mag = np.log(np.abs(f) + 1e-6)
            h, w = mag.shape
            cy, cx = h // 2, w // 2
            r = min(cy, cx) // 2
            center = mag[cy - r : cy + r, cx - r : cx + r].mean()
            outer = (mag.mean() * 4 - center) / 3.0
            hf_ratio = float(max(0.0, min(1.0, outer / (center + 1e-6))))
            # Edge variance (sharpness proxy)
            vy, vx = np.gradient(im_gray)
            edge_var = float(np.mean(vx**2 + vy**2))
            artifacts = []
            if hf_ratio > 0.8:
                artifacts.append(
                    {"name": "High frequency energy", "severity": "medium"}
                )
            if edge_var > 0.02:
                artifacts.append({"name": "Sharp edge transitions", "severity": "low"})
            result["artifacts"] = artifacts
        except Exception:
            pass

        # timeline placeholder (single frame)
        result["timeline"] = [{"t": 0.0, "score": prob_fake}]

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to image file")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Path to fine-tuned checkpoint directory (containing config.json, model.safetensors, etc.)",
    )
    parser.add_argument(
        "--out", default="", help="Optional path to write JSON result to"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress non-JSON output")
    parser.add_argument(
        "--explain", action="store_true", help="Return heatmap and simple artifacts"
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=float(os.environ.get("MODEL_TEMP", 1.5)),
        help="Temperature for softmax calibration (default 1.5)",
    )
    args = parser.parse_args()

    # Reduce noisy logs/warnings to keep stdout clean JSON
    if args.quiet:
        logging.getLogger("transformers").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")

    image_path = Path(args.input)
    if not image_path.exists():
        payload = {"error": f"input not found: {image_path}"}
        if args.out:
            Path(args.out).write_text(json.dumps(payload))
        print(json.dumps(payload))
        sys.exit(2)

    checkpoint_dir = Path(args.checkpoint) if args.checkpoint else None
    model, processor = load_model(checkpoint_dir)

    result = predict(
        model, processor, image_path, temperature=args.temp, explain=args.explain
    )
    if args.out:
        Path(args.out).write_text(json.dumps(result))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
