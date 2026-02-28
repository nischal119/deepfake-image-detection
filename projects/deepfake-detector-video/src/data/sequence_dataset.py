from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm


FrameSamplingMode = Literal["sliding", "uniform"]


@dataclass
class FrameRecord:
    video_id: str
    frame_path: Path
    frame_index: int
    label: int


class SequenceDataset(Dataset):
    """
    Build fixed-length frame sequences from pre-extracted frames and a master CSV.

    Expected layout (configurable but assumed by default):
      - frames_root: data/frames/
      - master CSV:  data/frames_master.csv

    The CSV should contain at least:
      - video_id: identifier for the source video
      - frame_path: path to the frame image, relative to frames_root OR absolute
      - frame_index: integer index within the video (0-based or 1-based; ordering is what matters)
      - label: 0 (real) or 1 (fake)

    Each dataset sample is:
      - frames: FloatTensor of shape (T, C, H, W)
      - label:  int (0/1)

    Sampling modes:
      - "sliding": sliding window of length `sequence_length` with stride `stride`
      - "uniform": one sequence per video, with `sequence_length` frames
                   sampled uniformly across the full clip
    """

    def __init__(
        self,
        frames_root: Path,
        master_csv: Path,
        sequence_length: int = 16,
        stride: int = 8,
        sampling_mode: FrameSamplingMode = "sliding",
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.frames_root = Path(frames_root)
        self.master_csv = Path(master_csv)
        self.sequence_length = sequence_length
        self.stride = stride
        self.sampling_mode = sampling_mode

        if transform is None:
            self.transform = transforms.Compose(
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
        else:
            self.transform = transform

        self._sequences: List[Tuple[List[FrameRecord], int]] = []
        self._build_index()

    def _build_index(self) -> None:
        df = pd.read_csv(self.master_csv)

        required_cols = {"video_id", "frame_path", "frame_index", "label"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"frames_master.csv is missing required columns: {sorted(missing)}"
            )

        grouped: Dict[str, List[FrameRecord]] = {}
        for _, row in df.iterrows():
            video_id = str(row["video_id"])
            label = int(row["label"])

            frame_path_val = str(row["frame_path"])
            frame_path = Path(frame_path_val)
            if not frame_path.is_absolute():
                frame_path = self.frames_root / frame_path

            frame_idx = int(row["frame_index"])

            rec = FrameRecord(
                video_id=video_id,
                frame_path=frame_path,
                frame_index=frame_idx,
                label=label,
            )
            grouped.setdefault(video_id, []).append(rec)

        # Sort frames within each video by frame_index
        for vid, frames in grouped.items():
            frames.sort(key=lambda r: r.frame_index)

        sequences: List[Tuple[List[FrameRecord], int]] = []

        if self.sampling_mode == "sliding":
            for vid, frames in grouped.items():
                if len(frames) < self.sequence_length:
                    continue  # too short to form a window
                label = frames[0].label
                for start in range(
                    0, len(frames) - self.sequence_length + 1, self.stride
                ):
                    window = frames[start : start + self.sequence_length]
                    sequences.append((window, label))
        elif self.sampling_mode == "uniform":
            for vid, frames in grouped.items():
                if not frames:
                    continue
                label = frames[0].label
                idxs = self._uniform_indices(len(frames), self.sequence_length)
                window = [frames[i] for i in idxs]
                sequences.append((window, label))
        else:
            raise ValueError(f"Unknown sampling_mode: {self.sampling_mode}")

        self._sequences = sequences
        print(
            f"SequenceDataset built from {len(grouped)} videos -> "
            f"{len(self._sequences)} sequences "
            f"(mode={self.sampling_mode}, T={self.sequence_length}, stride={self.stride})"
        )

    @staticmethod
    def _uniform_indices(num_frames: int, sequence_length: int) -> List[int]:
        if num_frames <= 0:
            return []
        if num_frames >= sequence_length:
            # spread indices roughly uniformly over [0, num_frames-1]
            lin = np.linspace(0, num_frames - 1, sequence_length)
            idxs = np.round(lin).astype(int)
        else:
            # not enough frames: repeat last frame
            base = list(range(num_frames))
            while len(base) < sequence_length:
                base.append(num_frames - 1)
            idxs = np.array(base[:sequence_length], dtype=int)
        return idxs.tolist()

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        frames_rec, label = self._sequences[idx]

        frames: List[Tensor] = []
        for rec in frames_rec:
            img = self._load_image(rec.frame_path)
            img_t = self.transform(img)
            frames.append(img_t)

        frames_tensor = torch.stack(frames, dim=0)  # (T, C, H, W)
        return frames_tensor, int(label)

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read frame image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class EmbeddingCache:
    """
    Helper for caching per-frame embeddings to disk to accelerate training.

    Typical usage:

        # 1) Build a SequenceDataset (returns images)
        ds = SequenceDataset(frames_root, master_csv, ...)

        # 2) Collect unique frame paths
        frame_paths = EmbeddingCache.collect_frame_paths(ds)

        # 3) Precompute embeddings with a backbone model
        cache = EmbeddingCache(cache_dir="data/frame_embeddings")
        cache.precompute(
            frame_paths=frame_paths,
            transform=ds.transform,
            backbone=backbone,
            device=device,
            batch_size=64,
        )

        # 4) Use SequenceEmbeddingDataset to read cached features instead of images

    The cache layout mirrors the frame paths, but with `.pt` files containing
    a single tensor per frame.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def collect_frame_paths(dataset: SequenceDataset) -> List[Path]:
        paths: List[Path] = []
        for seq, _ in dataset._sequences:
            for rec in seq:
                paths.append(rec.frame_path)
        # Unique, preserving order
        seen = set()
        unique: List[Path] = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    def _embedding_path(self, frame_path: Path) -> Path:
        # Store relative to cache_dir, mirroring original structure where possible.
        rel = frame_path
        # If frame_path is absolute, strip drive/root and treat remainder as relative.
        try:
            rel = frame_path.relative_to(frame_path.anchor)
        except ValueError:
            pass
        return self.cache_dir / rel.with_suffix(".pt")

    def has_embedding(self, frame_path: Path) -> bool:
        return self._embedding_path(frame_path).is_file()

    def load_embedding(self, frame_path: Path) -> Tensor:
        return torch.load(self._embedding_path(frame_path), map_location="cpu")

    def save_embedding(self, frame_path: Path, emb: Tensor) -> None:
        out_path = self._embedding_path(frame_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(emb.cpu(), out_path)

    def precompute(
        self,
        frame_paths: Sequence[Path],
        transform: transforms.Compose,
        backbone: torch.nn.Module,
        device: torch.device,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> None:
        """
        Compute embeddings for all given frame paths and cache them to disk.

        The backbone should map a batch of images (B, 3, H, W) to (B, D) features.
        """

        class _FrameDataset(Dataset):
            def __init__(self, paths: Sequence[Path], transform):
                self.paths = list(paths)
                self.transform = transform

            def __len__(self) -> int:
                return len(self.paths)

            def __getitem__(self, idx: int) -> Tuple[Tensor, Path]:
                p = self.paths[idx]
                img = cv2.imread(str(p))
                if img is None:
                    raise RuntimeError(f"Failed to read frame image: {p}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_t = self.transform(img)
                return img_t, p

        # Filter out frames we already have cached
        to_process = [p for p in frame_paths if not self.has_embedding(p)]
        if not to_process:
            print("EmbeddingCache: all frames already cached.")
            return

        ds = _FrameDataset(to_process, transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        backbone.eval()
        backbone.to(device)

        with torch.no_grad():
            for imgs, paths in tqdm(loader, desc="Caching frame embeddings"):
                imgs = imgs.to(device, non_blocking=True)
                feats = backbone(imgs)  # (B, D, ...) depending on backbone
                # Flatten any spatial dims if present
                if feats.dim() > 2:
                    feats = torch.flatten(feats, start_dim=1)
                for emb, p in zip(feats, paths):
                    self.save_embedding(Path(p), emb)


class SequenceEmbeddingDataset(Dataset):
    """
    Variant of SequenceDataset that reads precomputed frame embeddings
    (from EmbeddingCache) instead of raw images.

    Each sample returns:
      - features: FloatTensor of shape (T, D)
      - label:    int (0/1)
    """

    def __init__(
        self,
        base_dataset: SequenceDataset,
        cache: EmbeddingCache,
    ) -> None:
        self.base = base_dataset
        self.cache = cache

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        frames_rec, label = self.base._sequences[idx]
        feats: List[Tensor] = []
        for rec in frames_rec:
            emb = self.cache.load_embedding(rec.frame_path)
            feats.append(emb)
        features = torch.stack(feats, dim=0)  # (T, D)
        return features, int(label)


def example_dataloader() -> None:
    """
    Example usage of SequenceDataset and DataLoader.
    Not executed by default; call manually or copy this snippet.
    """
    frames_root = Path("data/frames")
    master_csv = Path("data/frames_master.csv")

    dataset = SequenceDataset(
        frames_root=frames_root,
        master_csv=master_csv,
        sequence_length=16,
        stride=8,
        sampling_mode="sliding",
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    for batch_idx, (frames, labels) in enumerate(loader):
        # frames: (B, T, C, H, W), labels: (B,)
        print("Batch", batch_idx, "frames shape:", frames.shape, "labels:", labels)
        break

