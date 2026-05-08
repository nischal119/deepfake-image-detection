from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


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

    Expected layout:
      - frames_root: data/frames/
      - master CSV:  data/frames_master.csv
    
    The CSV should contain: video_id, frame_path, frame_index, and label (0=real, 1=fake).
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

          
        for vid, frames in grouped.items():
            frames.sort(key=lambda r: r.frame_index)

        sequences: List[Tuple[List[FrameRecord], int]] = []

        if self.sampling_mode == "sliding":
            for vid, frames in grouped.items():
                if len(frames) < self.sequence_length:
                    continue    
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
              
            lin = np.linspace(0, num_frames - 1, sequence_length)
            idxs = np.round(lin).astype(int)
        else:
              
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

        frames_tensor = torch.stack(frames, dim=0)    
        return frames_tensor, int(label)

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read frame image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


