"""
Video-level scoring and evaluation for deepfake detection.

Implements:
  (a) Average of frame-level probabilities
  (b) Majority vote (per-frame hard preds)
  (c) Temporal model output (3D CNN / LSTM)
  (d) Ensemble: weighted average of (a) and (c)

Emits results/video_eval.csv with per-video predictions and metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm.auto import tqdm

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]


@dataclass
class VideoMetrics:
    auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def aggregate_to_video_level(
    frame_df: pd.DataFrame,
    temporal_df: Optional[pd.DataFrame] = None,
    ensemble_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Aggregate frame-level predictions to video-level scores.

    Expected frame_df columns (one row per frame):
      - video_id
      - frame_prob  (probability that frame is 'fake', in [0, 1])
      - label       (0 = real, 1 = fake)

    Optional temporal_df columns (one row per video):
      - video_id
      - temporal_prob  (probability from 3D CNN/LSTM, in [0, 1])

    Returns per-video DataFrame with columns:
      - video_id, label
      - prob_avg, pred_avg
      - prob_majority, pred_majority
      - prob_temporal, pred_temporal (if temporal_df provided)
      - prob_ensemble, pred_ensemble (if temporal_df provided)
    """
    required = {"video_id", "frame_prob", "label"}
    missing = required - set(frame_df.columns)
    if missing:
        raise ValueError(
            f"frame_df is missing required columns: {sorted(missing)} "
            "(expected video_id, frame_prob, label)"
        )

    grouped = frame_df.groupby("video_id")
    rows = []

    for vid, g in tqdm(grouped, desc="Aggregating videos"):
        labels = g["label"].values
        video_label = int(round(labels.mean()))
        frame_probs = g["frame_prob"].values.astype(float)

        # (a) Average of frame-level probabilities
        prob_avg = float(frame_probs.mean())
        pred_avg = int(prob_avg >= 0.5)

        # (b) Majority vote (per-frame hard classifications)
        hard_preds = (frame_probs >= 0.5).astype(int)
        prob_majority = float(hard_preds.mean())
        pred_majority = int(prob_majority >= 0.5)

        row = {
            "video_id": vid,
            "label": video_label,
            "prob_avg": prob_avg,
            "pred_avg": pred_avg,
            "prob_majority": prob_majority,
            "pred_majority": pred_majority,
        }
        rows.append(row)

    video_df = pd.DataFrame(rows)

    # (c) Temporal model output (3D CNN / LSTM)
    if temporal_df is not None:
        if not {"video_id", "temporal_prob"} <= set(temporal_df.columns):
            raise ValueError("temporal_df must have columns: video_id, temporal_prob")
        video_df = video_df.merge(
            temporal_df[["video_id", "temporal_prob"]], on="video_id", how="left"
        )
        video_df["prob_temporal"] = video_df["temporal_prob"].astype(float)
        video_df["pred_temporal"] = (video_df["prob_temporal"] >= 0.5).astype(int)

        # (d) Ensemble: weighted average of (a) and (c)
        w = float(ensemble_weight)
        video_df["prob_ensemble"] = np.where(
            video_df["prob_temporal"].notna(),
            w * video_df["prob_avg"] + (1.0 - w) * video_df["prob_temporal"],
            video_df["prob_avg"],
        )
        video_df["pred_ensemble"] = (video_df["prob_ensemble"] >= 0.5).astype(int)

    return video_df


def _compute_metrics(
    y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray
) -> VideoMetrics:
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return VideoMetrics(auc=auc, accuracy=acc, precision=prec, recall=rec, f1=f1)


def evaluate_video_predictions(
    video_df: pd.DataFrame,
    score_cols: Optional[Dict[str, str]] = None,
) -> Dict[str, VideoMetrics]:
    """
    Compute AUC, accuracy, precision, recall, F1 per scoring method.

    Parameters
    ----------
    video_df : DataFrame from aggregate_to_video_level
    score_cols : dict mapping method name -> prob column
        Default: avg, majority, temporal, ensemble (if present)

    Returns
    -------
    dict[str, VideoMetrics]
    """
    if "label" not in video_df.columns:
        raise ValueError("video_df must have 'label' column")

    if score_cols is None:
        score_cols = {"avg": "prob_avg", "majority": "prob_majority"}
        if "prob_temporal" in video_df.columns:
            score_cols["temporal"] = "prob_temporal"
        if "prob_ensemble" in video_df.columns:
            score_cols["ensemble"] = "prob_ensemble"

    y_true = video_df["label"].values.astype(int)
    metrics_by_method: Dict[str, VideoMetrics] = {}

    for name, prob_col in score_cols.items():
        if prob_col not in video_df.columns:
            continue
        y_score = video_df[prob_col].fillna(0.5).values.astype(float)
        y_pred = (y_score >= 0.5).astype(int)
        metrics_by_method[name] = _compute_metrics(y_true, y_score, y_pred)

    return metrics_by_method


def evaluate_from_csv(
    frame_preds_csv: Path,
    out_csv: Path,
    temporal_preds_csv: Optional[Path] = None,
    ensemble_weight: float = 0.5,
) -> Dict[str, VideoMetrics]:
    """
    Load frame-level predictions, aggregate to video level, compute metrics,
    and write per-video results to out_csv.

    Expected frame_preds_csv columns:
      - video_id, frame_index, frame_prob, label

    Optional temporal_preds_csv columns:
      - video_id, temporal_prob
    """
    frame_df = pd.read_csv(frame_preds_csv)
    temporal_df = pd.read_csv(temporal_preds_csv) if temporal_preds_csv else None

    video_df = aggregate_to_video_level(
        frame_df=frame_df,
        temporal_df=temporal_df,
        ensemble_weight=ensemble_weight,
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    video_df.to_csv(out_csv, index=False)
    print(f"Wrote per-video predictions to {out_csv}")

    metrics_by_method = evaluate_video_predictions(video_df)
    print("Per-video metrics:")
    for name, m in metrics_by_method.items():
        print(
            f"  [{name}] AUC={m.auc:.4f} Acc={m.accuracy:.4f} "
            f"Prec={m.precision:.4f} Rec={m.recall:.4f} F1={m.f1:.4f}"
        )
    return metrics_by_method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate frame-level predictions and evaluate at video level."
    )
    parser.add_argument(
        "--frame-preds",
        type=str,
        required=True,
        help="CSV with frame-level predictions (video_id, frame_index, frame_prob, label).",
    )
    parser.add_argument(
        "--temporal-preds",
        type=str,
        default=None,
        help="Optional CSV with temporal model predictions (video_id, temporal_prob).",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(PROJECT_ROOT / "results" / "video_eval.csv"),
        help="Output CSV for per-video predictions.",
    )
    parser.add_argument(
        "--ensemble-weight",
        type=float,
        default=0.5,
        help="Weight for frame-avg in ensemble: w*avg + (1-w)*temporal.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_preds_csv = Path(args.frame_preds).expanduser().resolve()
    temporal_preds_csv = (
        Path(args.temporal_preds).expanduser().resolve() if args.temporal_preds else None
    )
    out_csv = Path(args.out_csv).expanduser().resolve()

    print("Video scoring config:")
    print(f"  frame_preds:    {frame_preds_csv}")
    print(f"  temporal_preds: {temporal_preds_csv}")
    print(f"  out_csv:        {out_csv}")

    evaluate_from_csv(
        frame_preds_csv=frame_preds_csv,
        out_csv=out_csv,
        temporal_preds_csv=temporal_preds_csv,
        ensemble_weight=args.ensemble_weight,
    )


if __name__ == "__main__":
    main()
