
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
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

from src.models.video_lstm import ResNetLSTMVideoClassifier


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
            frame_idxs = list(range(frame_count))
            while len(frame_idxs) < self.clip_length:
                frame_idxs.append(max(0, frame_count - 1))
            frame_idxs = frame_idxs[: self.clip_length]
        else:
            lin = np.linspace(0, frame_count - 1, self.clip_length)
            frame_idxs = np.round(lin).astype(int).tolist()

        frames: List[torch.Tensor] = []
        for fi in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_t = self.transform(frame)
            frames.append(frame_t)

        cap.release()
        
        if len(frames) < self.clip_length:
             while len(frames) < self.clip_length:
                 frames.append(frames[-1] if frames else torch.zeros(3, self.frame_size, self.frame_size))

        clip = torch.stack(frames, dim=0)
        clip = clip.permute(1, 0, 2, 3)

        return clip, label


@dataclass
class TrainConfig:
    data_root: Path
    epochs: int = 15
    batch_size: int = 4
    lr: float = 1e-4
    val_fraction: float = 0.2
    num_workers: int = 0
    max_samples: Optional[int] = 400
    clip_length: int = 16
    frame_size: int = 112
    patience: int = 4
    pretrained: bool = True
    model_path: Path = PROJECT_ROOT / "models" / "video_lstm.pth"
    log_csv_path: Path = PROJECT_ROOT / "logs" / "video_lstm_training_log.csv"


def make_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    labeled_samples, unlabeled = collect_labeled_videos(cfg.data_root)
    print(f"Total labeled videos:   {len(labeled_samples)}")
    
    if cfg.max_samples is not None and len(labeled_samples) > cfg.max_samples:
        labeled_samples = labeled_samples[: cfg.max_samples]

    full_dataset = VideoClipDataset(
        labeled_samples,
        clip_length=cfg.clip_length,
        frame_size=cfg.frame_size,
    )
    
    val_size = int(len(full_dataset) * cfg.val_fraction)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

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
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print("Using device:", device)

    model = ResNetLSTMVideoClassifier(
        num_classes=2,
        pretrained=cfg.pretrained,
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_metric = -inf

    for epoch in range(1, cfg.epochs + 1):
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
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        append_log_row(cfg.log_csv_path, log_row)

        print(f"Epoch {epoch:02d} | val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")

        if metric > best_metric:
            best_metric = metric
            ensure_parent_dir(cfg.model_path)
            torch.save({"model_state": model.state_dict()}, cfg.model_path)
            print(f"  -> Saved model to {cfg.model_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else default_data_root()
    
    cfg = TrainConfig(
        data_root=data_root,
        epochs=args.epochs,
    )
    train(cfg)


if __name__ == "__main__":
    main()
