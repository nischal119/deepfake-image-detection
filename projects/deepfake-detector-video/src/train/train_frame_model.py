from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from math import inf
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from tqdm.auto import tqdm


HERE = Path(__file__).resolve()
# .../DeepFake Detector/projects/deepfake-detector-video/src/train/train_frame_model.py
REPO_ROOT = HERE.parents[4]
PROJECT_ROOT = HERE.parents[2]  # projects/deepfake-detector-video


def default_data_root() -> Path:
    """
    Default dataset root: `<repo-root>/deepfake-videos`.

    This assumes videos are laid out under:
      - deepfake-videos/DFD_original sequences/*.mp4
      - deepfake-videos/DFD_manipulated_sequences/DFD_manipulated_sequences/*.mp4
    """
    return REPO_ROOT / "deepfake-videos"


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


class ResNet50FrameClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        if pretrained:
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            backbone = models.resnet50(weights=None)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits


@dataclass
class TrainConfig:
    data_root: Path
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-4
    val_fraction: float = 0.2
    num_workers: int = 2
    max_samples: Optional[int] = 2000
    patience: int = 3
    pretrained: bool = True
    model_path: Path = PROJECT_ROOT / "models" / "frame_resnet50.pth"
    log_csv_path: Path = PROJECT_ROOT / "logs" / "frame_training_log.csv"


def make_dataloaders(cfg: TrainConfig):
    labeled_samples, unlabeled = collect_labeled_videos(cfg.data_root)
    print(f"Total labeled videos:   {len(labeled_samples)}")
    print(f"Total unlabeled videos: {len(unlabeled)}")

    if not labeled_samples:
        raise SystemExit(
            "No labeled videos were inferred from paths. "
            "Adjust LABEL_PATTERNS or organize your dataset "
            "into folders whose names contain 'real' / 'fake', etc."
        )

    print("Sample labeled paths:")
    for p, lab in labeled_samples[:5]:
        print(f"  {lab} -> {p}")

    if cfg.max_samples is not None and len(labeled_samples) > cfg.max_samples:
        labeled_samples = labeled_samples[: cfg.max_samples]
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

    val_size = int(len(full_dataset) * cfg.val_fraction)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Train size: {train_size}, Val size: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def run_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    *,
    phase: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float, float]:
    is_train = phase == "train"
    model.train(mode=is_train)

    running_loss = 0.0
    seen_samples = 0
    acc_sum = 0.0

    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    loop = tqdm(loader, desc=phase, leave=False)
    for images, targets in loop:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, targets)

            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        seen_samples += batch_size

        preds = torch.argmax(logits, dim=1)
        batch_acc = (preds == targets).float().mean().item()
        acc_sum += batch_acc * batch_size

        # For AUC (val only, but computing in both phases is cheap enough)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

        if seen_samples > 0:
            avg_loss_so_far = running_loss / seen_samples
            avg_acc_so_far = acc_sum / seen_samples
            loop.set_postfix(
                loss=f"{avg_loss_so_far:.4f}",
                acc=f"{avg_acc_so_far:.3f}",
            )

    avg_loss = running_loss / len(loader.dataset)
    acc = acc_sum / len(loader.dataset)

    y_true = np.concatenate(all_targets)
    y_score = np.concatenate(all_probs)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")

    return avg_loss, acc, auc


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_log_row(csv_path: Path, row: dict) -> None:
    ensure_parent_dir(csv_path)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train(cfg: TrainConfig) -> None:
    train_loader, val_loader = make_dataloaders(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = ResNet50FrameClassifier(num_classes=2, pretrained=cfg.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_metric = -inf
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        # Train phase
        model.train()
        train_loss, train_acc, train_auc = run_epoch(
            train_loader,
            model,
            criterion,
            device,
            phase="train",
            optimizer=optimizer,
        )

        # Val phase
        with torch.no_grad():
            val_loss, val_acc, val_auc = run_epoch(
                val_loader, model, criterion, device, phase="val", optimizer=None
            )

        # Choose metric for early stopping / checkpointing
        metric = val_auc if not np.isnan(val_auc) else val_acc

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_auc": train_auc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
        }
        append_log_row(cfg.log_csv_path, log_row)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, train_auc={train_auc:.3f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}, val_auc={val_auc:.3f}"
        )

        if metric > best_metric:
            best_metric = metric
            best_state = {
                "model_state": model.state_dict(),
                "config": asdict(cfg),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
            }
            epochs_no_improve = 0
            ensure_parent_dir(cfg.model_path)
            torch.save(best_state, cfg.model_path)
            print(f"  -> Saved new best model to {cfg.model_path} (metric={metric:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= cfg.patience:
            print(
                f"Early stopping triggered (patience={cfg.patience}, "
                f"best_metric={best_metric:.4f})."
            )
            break


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a frame-based ResNet-50 deepfake classifier."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory containing videos (default: <repo-root>/deepfake-videos)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Maximum number of videos to use (None for all)",
    )
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet-pretrained weights for ResNet-50.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "frame_resnet50.pth"),
        help="Path to save best model checkpoint.",
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default=str(PROJECT_ROOT / "logs" / "frame_training_log.csv"),
        help="Path to CSV log file.",
    )

    args = parser.parse_args()

    data_root = (
        default_data_root()
        if args.data_root is None
        else Path(args.data_root).expanduser().resolve()
    )
    if not data_root.exists():
        raise SystemExit(f"Data root does not exist: {data_root}")

    cfg = TrainConfig(
        data_root=data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        patience=args.patience,
        pretrained=not args.no_pretrained,
        model_path=Path(args.model_path).expanduser().resolve(),
        log_csv_path=Path(args.log_csv).expanduser().resolve(),
    )
    return cfg


def main() -> None:
    cfg = parse_args()
    print("Training configuration:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    train(cfg)


if __name__ == "__main__":
    main()

