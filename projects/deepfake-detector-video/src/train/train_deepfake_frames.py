from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from tqdm.auto import tqdm


def default_data_root() -> Path:
    """
    Infer the dataset root as `<repo-root>/deepfake-videos`.

    This assumes the following layout:
    - repo root: DeepFake Detector/
    - dataset:   DeepFake Detector/deepfake-videos/
    - this file: DeepFake Detector/projects/deepfake-detector-video/src/train/train_deepfake_frames.py
    """
    here = Path(__file__).resolve()
    # .../DeepFake Detector/projects/deepfake-detector-video/src/train/train_deepfake_frames.py
    # parents[0]=train, [1]=src, [2]=deepfake-detector-video, [3]=projects, [4]=DeepFake Detector
    repo_root = here.parents[4]
    return repo_root / "deepfake-videos"


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# Map lowercase substring -> numeric label
LABEL_PATTERNS = {
    "real": 0,
    "original": 0,
    "authentic": 0,
    "fake": 1,
    "manipulated": 1,
    "deepfake": 1,
}


def infer_label_from_path(path: Path) -> Optional[int]:
    s = str(path).lower()
    for key, label in LABEL_PATTERNS.items():
        if key in s:
            return label
    return None


def collect_labeled_videos(root: Path) -> Tuple[List[Tuple[Path, int]], List[Path]]:
    video_paths = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]

    labeled: List[Tuple[Path, int]] = []
    unlabeled: List[Path] = []

    for p in tqdm(sorted(video_paths), desc="Indexing videos"):
        label = infer_label_from_path(p)
        if label is None:
            unlabeled.append(p)
        else:
            labeled.append((p, label))

    return labeled, unlabeled


class SingleFrameVideoDataset(Dataset):
    """
    Treat each video as a sample and use a single representative frame
    (the middle frame) as input to an image classifier.
    """

    def __init__(self, samples: Sequence[Tuple[Path, int]], transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        label = int(label)

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

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            frame = self.transform(frame)

        return frame, label


class ResNetFrameClassifier(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Use ImageNet-pretrained ResNet-18 (requires network + valid SSL certs).
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits


def run_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    *,
    epoch: int = 0,
    total_epochs: int = 0,
    phase: str = "train",
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    running_loss = 0.0
    seen_samples = 0
    acc_sum = 0.0

    desc = phase
    if epoch and total_epochs:
        desc = f"{phase} {epoch}/{total_epochs}"

    loop = tqdm(loader, desc=desc, leave=False)
    for images, targets in loop:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        seen_samples += batch_size

        preds = torch.argmax(logits, dim=1)
        batch_acc = (preds == targets).float().mean().item()
        acc_sum += batch_acc * batch_size

        # Live progress update
        if seen_samples > 0:
            avg_loss_so_far = running_loss / seen_samples
            avg_acc_so_far = acc_sum / seen_samples
            loop.set_postfix(
                loss=f"{avg_loss_so_far:.4f}",
                acc=f"{avg_acc_so_far:.3f}",
            )

    avg_loss = running_loss / len(loader.dataset)
    acc = acc_sum / len(loader.dataset)
    return avg_loss, acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a simple frame-based deepfake classifier on videos in deepfake-videos/"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory containing videos (default: <repo-root>/deepfake-videos)",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Maximum number of videos to use (None for all)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)

    args = parser.parse_args()

    if args.data_root is None:
        data_root = default_data_root()
    else:
        data_root = Path(args.data_root).expanduser().resolve()

    print(f"Using data root: {data_root}")
    if not data_root.exists():
        raise SystemExit(f"Data root does not exist: {data_root}")

    labeled_samples, unlabeled = collect_labeled_videos(data_root)
    print(f"Total labeled videos:   {len(labeled_samples)}")
    print(f"Total unlabeled videos: {len(unlabeled)}")

    if not labeled_samples:
        raise SystemExit(
            "No labeled videos were inferred from paths. "
            "Adjust LABEL_PATTERNS in this script or organize your dataset "
            "into folders whose names contain 'real' / 'fake', etc."
        )

    print("Sample labeled paths:")
    for p, lab in labeled_samples[:5]:
        print(f"  {lab} -> {p}")

    if args.max_samples is not None and len(labeled_samples) > args.max_samples:
        labeled_samples = labeled_samples[: args.max_samples]
        print(f"Subsampled to {len(labeled_samples)} labeled videos.")

    image_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    full_dataset = SingleFrameVideoDataset(labeled_samples, transform=image_transform)
    print(f"Final dataset size: {len(full_dataset)}")

    val_size = int(len(full_dataset) * args.val_fraction)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Train size: {train_size}, Val size: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = ResNetFrameClassifier(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            train_loader,
            model,
            criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            phase="train",
        )
        val_loss, val_acc = run_epoch(
            val_loader,
            model,
            criterion,
            optimizer=None,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            phase="val",
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )


if __name__ == "__main__":
    main()

