"""
Model optimization and export for deployment.

Converts trained models to TorchScript or ONNX for faster inference.
Supports FP16 conversion and dynamic quantization.

Latency / trade-offs (typical on CPU, single-threaded):
- Original (FP32):     baseline
- TorchScript (FP32):  ~10–20% faster (fewer Python overheads)
- TorchScript (FP16):  ~30–50% faster on GPU; may be slower on CPU
- Dynamic quant (int8): ~2–4× faster on CPU; slight accuracy drop (<1%)
- ONNX (FP32):        portable; often 1.2–1.5× faster with ONNX Runtime
- ONNX + FP16:        smaller file; GPU speedup

Trade-offs:
- FP16: smaller model, faster on GPU; possible numerical instability, slower on some CPUs
- Dynamic quantization: CPU speedup, no calibration needed; small accuracy loss on quantized layers
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Optional

import torch
from torchvision import models, transforms

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DEPLOY_DIR = PROJECT_ROOT / "deploy"


def _build_frame_model(ckpt_path: Path) -> torch.nn.Module:
    """Build ResNet50 frame classifier from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)

    backbone = models.resnet50(weights=None)
    in_features = backbone.fc.in_features
    backbone.fc = torch.nn.Identity()

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.classifier = torch.nn.Linear(in_features, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.backbone(x))

    model = _Model()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def export_torchscript(
    model: torch.nn.Module,
    output_path: Path,
    example_input: torch.Tensor,
    optimize: bool = True,
) -> Path:
    """Export model to TorchScript (traced)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.trace(model, example_input, optimize=optimize)
    traced.save(str(output_path))
    return output_path


def export_onnx(
    model: torch.nn.Module,
    output_path: Path,
    example_input: torch.Tensor,
    opset_version: int = 14,
    dynamic_axes: Optional[dict] = None,
) -> Path:
    """Export model to ONNX."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = dynamic_axes or {"input": {0: "batch"}, "output": {0: "batch"}}
    torch.onnx.export(
        model,
        example_input,
        str(output_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    return output_path


def apply_fp16(model: torch.nn.Module) -> torch.nn.Module:
    """Convert model to FP16 (half precision)."""
    return model.half()


def apply_dynamic_quantization(model: torch.nn.Module) -> torch.nn.Module:
    """Apply dynamic quantization to linear/conv layers (int8 activations, fp32 compute in some backends)."""
    return torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8,
    )


def export_frame_model(
    checkpoint: Path = MODELS_DIR / "frame_resnet50.pth",
    output_dir: Path = DEPLOY_DIR,
    formats: str = "torchscript,onnx",
    fp16: bool = False,
    quantize: bool = False,
    image_size: int = 224,
) -> dict:
    """
    Export frame classifier to TorchScript and/or ONNX.

    Returns dict of {format: path} for exported files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _build_frame_model(Path(checkpoint))

    example = torch.randn(1, 3, image_size, image_size)
    if fp16:
        model = apply_fp16(model)
        example = example.half()
        suffix = "_fp16"
    elif quantize:
        model = apply_dynamic_quantization(model)
        suffix = "_q8"
    else:
        suffix = ""

    results = {}
    fmt_list = [f.strip() for f in formats.lower().split(",")]

    if "torchscript" in fmt_list:
        out = output_dir / f"frame_resnet50{suffix}.pt"
        export_torchscript(model, out, example)
        results["torchscript"] = out

    if "onnx" in fmt_list and not quantize:
        model_onnx = _build_frame_model(Path(checkpoint))
        example_onnx = torch.randn(1, 3, image_size, image_size)
        if fp16:
            model_onnx = model_onnx.half()
            example_onnx = example_onnx.half()
        out = output_dir / f"frame_resnet50_onnx{suffix}.onnx"
        export_onnx(model_onnx, out, example_onnx)
        results["onnx"] = out

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export trained models to TorchScript/ONNX with optional optimization."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(MODELS_DIR / "frame_resnet50.pth"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEPLOY_DIR),
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="torchscript,onnx",
        help="Comma-separated: torchscript, onnx",
    )
    parser.add_argument("--fp16", action="store_true", help="Export in FP16")
    parser.add_argument("--quantize", action="store_true", help="Apply dynamic int8 quantization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = export_frame_model(
        checkpoint=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        formats=args.formats,
        fp16=args.fp16,
        quantize=args.quantize,
    )
    print("Exported:")
    for fmt, path in results.items():
        print(f"  {fmt}: {path}")


if __name__ == "__main__":
    main()
