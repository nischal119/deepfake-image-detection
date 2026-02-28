"""
Fairness evaluation for deepfake detection models.

Computes performance (accuracy, FPR, FNR) per demographic group and produces
a fairness report with tables and charts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]

DEMOGRAPHIC_COLS = ["gender", "skin_tone", "age_group"]


@dataclass
class GroupMetrics:
    """Per-group performance metrics."""

    group_name: str
    n_samples: int
    n_positive: int  # true label = 1 (fake)
    n_negative: int  # true label = 0 (real)
    accuracy: float
    fpr: float  # false positive rate: P(pred=1|true=0)
    fnr: float  # false negative rate: P(pred=0|true=1)
    precision: float
    recall: float
    f1: float


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> GroupMetrics:
    """Compute metrics for a single group."""
    n = len(y_true)
    n_positive = int((y_true == 1).sum())
    n_negative = int((y_true == 0).sum())

    if n == 0:
        return GroupMetrics(
            group_name="",
            n_samples=0,
            n_positive=0,
            n_negative=0,
            accuracy=0.0,
            fpr=0.0,
            fnr=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
        )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / n
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return GroupMetrics(
        group_name="",
        n_samples=n,
        n_positive=n_positive,
        n_negative=n_negative,
        accuracy=accuracy,
        fpr=fpr,
        fnr=fnr,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def load_and_merge(
    predictions_csv: Path,
    demographics_csv: Optional[Path] = None,
    id_col: str = "video_id",
    label_col: str = "label",
    pred_col: Optional[str] = None,
    prob_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load predictions and optionally merge with demographics.

    predictions_csv must have: id_col, label_col, and either pred_col or prob_col.
    If prob_col given, pred = (prob >= 0.5).
    demographics_csv must have: id_col + at least one of gender, skin_tone, age_group.
    """
    df = pd.read_csv(predictions_csv)

    if pred_col is None and prob_col is not None:
        df["pred"] = (df[prob_col].astype(float) >= 0.5).astype(int)
    elif pred_col is not None:
        df["pred"] = df[pred_col].astype(int)
    else:
        raise ValueError("Provide either pred_col or prob_col")

    df["label"] = df[label_col].astype(int)
    df["id"] = df[id_col].astype(str)

    if demographics_csv is not None:
        demo = pd.read_csv(demographics_csv)
        demo["id"] = demo[id_col].astype(str)
        for c in DEMOGRAPHIC_COLS:
            if c in demo.columns and c not in df.columns:
                df = df.merge(demo[["id", c]], on="id", how="left")

    return df


def compute_per_group_metrics(
    df: pd.DataFrame,
    demographic_col: str,
    label_col: str = "label",
    pred_col: str = "pred",
) -> Dict[str, GroupMetrics]:
    """Compute metrics for each value of demographic_col."""
    y_true = df[label_col].values
    y_pred = df[pred_col].values
    groups = df[demographic_col].fillna("unknown").astype(str)

    results: Dict[str, GroupMetrics] = {}
    for g in groups.unique():
        mask = groups == g
        n = mask.sum()
        if n == 0:
            continue
        m = _compute_metrics(y_true[mask], y_pred[mask])
        m.group_name = g
        results[g] = m
    return results


def run_fairness_evaluation(
    df: pd.DataFrame,
    output_dir: Path,
    demographic_cols: Optional[List[str]] = None,
) -> Dict[str, Dict[str, GroupMetrics]]:
    """
    Compute fairness metrics per demographic attribute and write report.

    Returns nested dict: {demographic_col: {group_value: GroupMetrics}}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    demographic_cols = demographic_cols or [c for c in DEMOGRAPHIC_COLS if c in df.columns]

    if not demographic_cols:
        raise ValueError(
            f"No demographic columns found. Expected one of {DEMOGRAPHIC_COLS} in dataframe."
        )

    all_results: Dict[str, Dict[str, GroupMetrics]] = {}

    for col in demographic_cols:
        if col not in df.columns:
            continue
        metrics_by_group = compute_per_group_metrics(df, col)
        all_results[col] = metrics_by_group

        # Table
        rows = []
        for g, m in metrics_by_group.items():
            rows.append({
                "group": g,
                "n_samples": m.n_samples,
                "n_positive": m.n_positive,
                "n_negative": m.n_negative,
                "accuracy": round(m.accuracy, 4),
                "fpr": round(m.fpr, 4),
                "fnr": round(m.fnr, 4),
                "precision": round(m.precision, 4),
                "recall": round(m.recall, 4),
                "f1": round(m.f1, 4),
            })
        table_df = pd.DataFrame(rows)
        csv_path = output_dir / f"fairness_{col}.csv"
        table_df.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}")

        # Chart: accuracy, FPR, FNR by group
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        groups = list(metrics_by_group.keys())
        accs = [metrics_by_group[g].accuracy for g in groups]
        fprs = [metrics_by_group[g].fpr for g in groups]
        fnrs = [metrics_by_group[g].fnr for g in groups]

        axes[0].bar(groups, accs, color="steelblue", alpha=0.8)
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title(f"Accuracy by {col}")
        axes[0].tick_params(axis="x", rotation=45)

        axes[1].bar(groups, fprs, color="coral", alpha=0.8)
        axes[1].set_ylabel("FPR")
        axes[1].set_title(f"False Positive Rate by {col}")
        axes[1].tick_params(axis="x", rotation=45)

        axes[2].bar(groups, fnrs, color="seagreen", alpha=0.8)
        axes[2].set_ylabel("FNR")
        axes[2].set_title(f"False Negative Rate by {col}")
        axes[2].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        fig_path = output_dir / f"fairness_{col}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Wrote {fig_path}")

    # Summary CSV (all attributes)
    summary_rows = []
    for col, metrics_by_group in all_results.items():
        for g, m in metrics_by_group.items():
            summary_rows.append({
                "demographic": col,
                "group": g,
                "n_samples": m.n_samples,
                "accuracy": round(m.accuracy, 4),
                "fpr": round(m.fpr, 4),
                "fnr": round(m.fnr, 4),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "fairness_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    # Mitigation suggestions
    suggestions = _generate_mitigation_suggestions(all_results)
    sugg_path = output_dir / "fairness_mitigation_suggestions.txt"
    sugg_path.write_text(suggestions, encoding="utf-8")
    print(f"Wrote {sugg_path}")

    return all_results


def _generate_mitigation_suggestions(
    all_results: Dict[str, Dict[str, GroupMetrics]],
) -> str:
    """Generate mitigation suggestions based on fairness gaps."""
    lines = [
        "=" * 60,
        "FAIRNESS EVALUATION – MITIGATION SUGGESTIONS",
        "=" * 60,
        "",
    ]

    for col, metrics_by_group in all_results.items():
        if len(metrics_by_group) < 2:
            continue

        accs = [m.accuracy for m in metrics_by_group.values()]
        fprs = [m.fpr for m in metrics_by_group.values()]
        fnrs = [m.fnr for m in metrics_by_group.values()]
        acc_gap = max(accs) - min(accs) if accs else 0
        fpr_gap = max(fprs) - min(fprs) if fprs else 0
        fnr_gap = max(fnrs) - min(fnrs) if fnrs else 0

        lines.append(f"[{col}]")
        lines.append(f"  Accuracy gap: {acc_gap:.3f}")
        lines.append(f"  FPR gap:      {fpr_gap:.3f}")
        lines.append(f"  FNR gap:      {fnr_gap:.3f}")

        if acc_gap > 0.05 or fpr_gap > 0.05 or fnr_gap > 0.05:
            lines.append("  Suggested mitigations:")
            lines.append("    1. Reweighing: upweight underrepresented groups in the loss.")
            lines.append("    2. Data augmentation: oversample minority groups or augment")
            lines.append("       to balance demographics.")
            lines.append("    3. Stratified sampling: ensure balanced groups in train/val.")
            lines.append("    4. Adversarial debiasing: add a group-predictor adversary.")
            lines.append("    5. Threshold tuning: set per-group decision thresholds.")
        else:
            lines.append("  Gaps are small; monitor in production.")
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fairness evaluation on deepfake detection predictions."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="CSV with video_id, label, and pred or prob.",
    )
    parser.add_argument(
        "--demographics",
        type=str,
        default=None,
        help="CSV with video_id and demographic columns (gender, skin_tone, age_group).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "fairness"),
    )
    parser.add_argument("--id-col", type=str, default="video_id")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--pred-col", type=str, default=None)
    parser.add_argument("--prob-col", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = Path(args.predictions).expanduser().resolve()
    demo_path = Path(args.demographics).expanduser().resolve() if args.demographics else None
    out_dir = Path(args.output_dir).expanduser().resolve()

    if not pred_path.exists():
        raise SystemExit(f"Predictions file not found: {pred_path}")

    df = load_and_merge(
        predictions_csv=pred_path,
        demographics_csv=demo_path,
        id_col=args.id_col,
        label_col=args.label_col,
        pred_col=args.pred_col,
        prob_col=args.prob_col,
    )

    run_fairness_evaluation(df, output_dir=out_dir)


if __name__ == "__main__":
    main()
