# Deploy: Model Optimization and Inference

## Export (optimize_export.py)

Converts the trained frame classifier to deployment-friendly formats:

```bash
# Export to TorchScript and ONNX (FP32)
python -m src.deploy.optimize_export

# Export with FP16 (smaller, faster on GPU)
python -m src.deploy.optimize_export --fp16

# Export with dynamic int8 quantization (faster on CPU)
python -m src.deploy.optimize_export --quantize
```

## Inference (infer_video.py)

Runs inference on a video and returns JSON:

```bash
python -m src.deploy.infer_video path/to/video.mp4
python -m src.deploy.infer_video video.mp4 --model deploy/frame_resnet50.pt --output result.json
```

## Latency and Trade-offs

| Format | Typical latency vs baseline | Notes |
|--------|-----------------------------|-------|
| **Original (FP32)** | 1.0× | Baseline; standard PyTorch |
| **TorchScript (FP32)** | 0.85–0.9× | 10–20% faster; fewer Python overheads |
| **TorchScript (FP16)** | 0.5–0.7× on GPU | 30–50% GPU speedup; may be slower on CPU |
| **Dynamic quant (int8)** | 0.25–0.5× on CPU | 2–4× CPU speedup; &lt;1% accuracy drop |
| **ONNX (FP32)** | 0.7–0.85× | Portable; often faster with ONNX Runtime |
| **ONNX + FP16** | 0.5–0.7× on GPU | Smaller file; GPU-oriented |

### Trade-offs

- **FP16**: Smaller model, faster on GPU; possible numerical instability; often slower on CPU without hardware support.
- **Dynamic quantization**: No calibration; only quantizes linear/conv; small accuracy loss on quantized layers.
- **ONNX**: Cross-framework; requires `onnxruntime` for inference; good for serving.
- **TorchScript**: PyTorch-native; no extra runtime; good for mobile/edge.

## Output Layout

Exports go to `projects/deepfake-detector-video/deploy/`:

```
deploy/
├── frame_resnet50.pt         # TorchScript FP32
├── frame_resnet50_fp16.pt    # TorchScript FP16
├── frame_resnet50_q8.pt      # TorchScript quantized
├── frame_resnet50_onnx.onnx  # ONNX FP32
└── ...
```
