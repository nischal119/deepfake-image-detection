"""Train R3D-18 3D video classifier on deepfake clips."""

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
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm.auto import tqdm

from src.models.video_r3d import R3D18VideoClassifier


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[4]
PROJECT_ROOT = HERE.parents[2]


def default_data_root() -> Path:
    return REPO_ROOT / "deepfake-videos"


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
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


class VideoClipDataset(Dataset):
    """
    Sample fixed-length clips (N frames) from videos for 3D CNN input.

    Each sample returns (C, T, H, W) so that DataLoader batches to (B, C, T, H, W)
    for R3D input.
    """

    def __init__(
        self,
        samples: Sequence[Tuple[Path, int]],
        clip_length: int = 16,
        frame_size: int = 112,
        transform=None,
    ) -> None:
        self.samples = list(samples)
        self.clip_length = clip_length
        self.frame_size = frame_size

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((frame_size, frame_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        label = int(label)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count < self.clip_length:
            # Repeat last frame to fill
            frame_idxs = list(range(frame_count))
            while len(frame_idxs) < self.clip_length:
                frame_idxs.append(max(0, frame_count - 1))
            frame_idxs = frame_idxs[: self.clip_length]
        else:
            # Sample uniformly from the video
            lin = np.linspace(0, frame_count - 1, self.clip_length)
            frame_idxs = np.round(lin).astype(int).tolist()

        frames: List[torch.Tensor] = []
        for fi in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to read frame {fi} from {path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_t = self.transform(frame)
            frames.append(frame_t)

        cap.release()

        # Stack: (T, C, H, W) -> permute to (C, T, H, W) for R3D
        clip = torch.stack(frames, dim=0)  # (T, C, H, W)
        clip = clip.permute(1, 0, 2, 3)   # (C, T, H, W)

        return clip, label


@dataclass
class TrainConfig:
    data_root: Path
    epochs: int = 15
    batch_size: int = 8
    lr: float = 1e-4
    val_fraction: float = 0.2
    num_workers: int = 2
    max_samples: Optional[int] = 400
    clip_length: int = 16
    frame_size: int = 112
    patience: int = 4
    pretrained: bool = True
    model_path: Path = PROJECT_ROOT / "models" / "video_r3d18.pth"
    log_csv_path: Path = PROJECT_ROOT / "logs" / "video_r3d18_training_log.csv"


def make_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    labeled_samples, unlabeled = collect_labeled_videos(cfg.data_root)
    print(f"Total labeled videos:   {len(labeled_samples)}")
    print(f"Total unlabeled videos: {len(unlabeled)}")

    if not labeled_samples:
        raise SystemExit(
            "No labeled videos were inferred. Adjust LABEL_PATTERNS or organize dataset."
        )

    print("Sample labeled paths:")
    for p, lab in labeled_samples[:5]:
        print(f"  {lab} -> {p}")

    if cfg.max_samples is not None and len(labeled_samples) > cfg.max_samples:
        labeled_samples = labeled_samples[: cfg.max_samples]
        print(f"Subsampled to {len(labeled_samples)} labeled videos.")

    full_dataset = VideoClipDataset(
        labeled_samples,
        clip_length=cfg.clip_length,
        frame_size=cfg.frame_size,
    )
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
    for clips, targets in loop:
        clips = clips.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(clips)
            loss = criterion(logits, targets)

            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_size = clips.size(0)
        running_loss += loss.item() * batch_size
        seen_samples += batch_size

        preds = torch.argmax(logits, dim=1)
        batch_acc = (preds == targets).float().mean().item()
        acc_sum += batch_acc * batch_size

        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

        if seen_samples > 0:
            loop.set_postfix(
                loss=f"{running_loss / seen_samples:.4f}",
                acc=f"{acc_sum / seen_samples:.3f}",
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

    model = R3D18VideoClassifier(
        num_classes=2,
        pretrained=cfg.pretrained,
        input_clip_length=cfg.clip_length,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_metric = -inf
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss, train_acc, train_auc = run_epoch(
            train_loader, model, criterion, device,
            phase="train", optimizer=optimizer,
        )
        with torch.no_grad():
            val_loss, val_acc, val_auc = run_epoch(
                val_loader, model, criterion, device,
                phase="val", optimizer=None,
            )

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
            epochs_no_improve = 0
            ensure_parent_dir(cfg.model_path)
            torch.save({
                "model_state": model.state_dict(),
                "config": asdict(cfg),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
            }, cfg.model_path)
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
        description="Train R3D-18 3D video deepfake classifier."
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=400)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--frame-size", type=int, default=112)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "video_r3d18.pth"),
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default=str(PROJECT_ROOT / "logs" / "video_r3d18_training_log.csv"),
    )

    args = parser.parse_args()
    data_root = default_data_root() if args.data_root is None else Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        raise SystemExit(f"Data root does not exist: {data_root}")

    return TrainConfig(
        data_root=data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        clip_length=args.clip_length,
        frame_size=args.frame_size,
        patience=args.patience,
        pretrained=not args.no_pretrained,
        model_path=Path(args.model_path).expanduser().resolve(),
        log_csv_path=Path(args.log_csv).expanduser().resolve(),
    )


def main() -> None:
    cfg = parse_args()
    print("Training configuration:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    train(cfg)


if __name__ == "__main__":
    main()
