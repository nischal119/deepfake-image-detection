## DeepFake Detector – Video Module

This subproject extends the main DeepFake Detector to **video input**, handling raw videos, frame extraction, training, and evaluation of video-based deepfake models.

### Project Structure

- `data/` – High-level data directory
  - `data/raw_videos/` – Original video files (e.g., MP4, AVI)
  - `data/frames/` – Extracted frames or face crops from videos
- `notebooks/` – Exploratory analysis, experiments, and prototyping
- `src/` – Python source code
  - `src/data/` – Datasets, transforms, and data-loading utilities
  - `src/models/` – Model definitions (e.g., 3D CNN, ViT, TimeSformer, PyTorchVideo models)
  - `src/train/` – Training loops, schedulers, and experiment runners
  - `src/eval/` – Evaluation scripts, metrics, and report generation
  - `src/utils/` – Shared helpers (logging, config, tracking, etc.)
- `scripts/` – CLI utilities for data prep, training, and inference
- `deploy/` – Exported artifacts and integration hooks (e.g., model export, worker entrypoints)

---

### Prerequisites

- Python 3.9+ (recommended to match the main project)
- A working GPU setup with CUDA if you plan to train/video-infer at scale
- `ffmpeg` installed on the system (for video decoding)

---

### Setup

From the repository root:

```bash
cd "DeepFake Detector"
cd projects/deepfake-detector-video

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

### Typical Workflow

1. **Organize raw videos**
   - Place input videos into `data/raw_videos/`.
   - Keep a CSV/JSON (in `data/`) with labels and metadata if needed (e.g., `video_id`, `label`, `source_dataset`).

2. **Extract frames (or clips)**
   - Implement or use a script in `scripts/` (e.g., `extract_frames.py`) that:
     - Reads videos from `data/raw_videos/`
     - Uses `ffmpeg-python`, `opencv-python`, and `albumentations` to:
       - Decode frames
       - Optionally detect faces (e.g., via `facenet-pytorch`)
       - Save processed frames/clips to `data/frames/`

3. **Prepare datasets**
   - In `src/data/`, implement dataset classes that:
     - Read frame paths / clip indices and labels from CSV/JSON
     - Apply spatial + temporal augmentations using:
       - `albumentations` (image-level)
       - `torchvision` / `pytorchvideo` transforms (video-level)

4. **Define models**
   - In `src/models/`, define architectures such as:
     - `pytorchvideo` backbones (e.g., X3D, SlowFast)
     - Transformer-based video models using `timm` or custom modules
   - Ensure models expose a simple interface: `(video_batch) -> logits`.

5. **Train**
   - In `src/train/`, implement a training script (e.g., `train_video.py`) that:
     - Loads datasets and dataloaders
     - Builds a model and optimizer (PyTorch)
     - Logs metrics and checkpoints (e.g., via CSV, TensorBoard, or any preferred logger)
   - Add a thin CLI wrapper in `scripts/` (e.g., `run_training.sh` or `train_video.sh`) to simplify running experiments.

6. **Evaluate**
   - In `src/eval/`, implement evaluation scripts that:
     - Load a trained checkpoint
     - Run inference over validation/test splits
     - Compute metrics (e.g., accuracy, ROC-AUC, F1) using `scikit-learn`, `pandas`, and `numpy`
     - Optionally export per-video or per-frame scores to CSV for analysis.

7. **Deploy / Integrate with main app**
   - In `deploy/`, add:
     - Export scripts to save models in a reproducible format (e.g., TorchScript, state dict + config).
     - Integration hooks or worker entrypoints that:
       - Connect to your main backend (e.g., via Celery + Redis, or direct API calls)
       - Read jobs from the queue
       - Run video inference with the trained model
       - Persist results to the database (`sqlalchemy`, `psycopg2-binary`) to be consumed by the frontend.

---

### Running (High-Level)

Below is a suggested minimal sequence once you have scripts in place:

```bash
# 1. Activate environment
cd "DeepFake Detector"
cd projects/deepfake-detector-video
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 2. Extract frames from raw videos (example script)
python scripts/extract_frames.py \
  --input-dir data/raw_videos \
  --output-dir data/frames \
  --fps 5

# 3. Train a video model
python -m src.train.train_video \
  --config configs/base_video.yaml

# 4. Evaluate a checkpoint
python -m src.eval.evaluate_video \
  --checkpoint-path checkpoints/best.ckpt
```

Adapt these commands to the actual scripts and configs you create.

---

### Notes

- This directory is intentionally **model-agnostic**: you can plug in different video architectures and datasets.
- Use the `notebooks/` folder for quick experiments, EDA on labels, and visualization of temporal predictions.
- For production, follow the main repository’s conventions for logging, configuration, and deployment so that the video module integrates cleanly with the existing image-based pipeline.

