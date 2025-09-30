#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

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


@torch.inference_mode()
def predict(model, processor, image_path: Path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]

    # Assume label mapping: index 0 -> Real, 1 -> Fake as in training script
    prob_fake = float(probs[1].item())
    prob_real = float(probs[0].item())

    if prob_fake > 0.7:
        verdict = "likely_fake"
    elif prob_fake < 0.3:
        verdict = "likely_real"
    else:
        verdict = "inconclusive"

    return {
        "score": prob_fake,
        "verdict": verdict,
        "probs": {"real": prob_real, "fake": prob_fake},
    }


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

    result = predict(model, processor, image_path)
    if args.out:
        Path(args.out).write_text(json.dumps(result))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
