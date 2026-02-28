"""Evaluation utilities for video deepfake detection."""

from .video_scoring import (
    VideoMetrics,
    aggregate_to_video_level,
    evaluate_video_predictions,
    evaluate_from_csv,
)
from .robustness_tests import (
    run_robustness_tests,
    create_degraded_variants,
    DEGRADATION_FUNCS,
)

__all__ = [
    "VideoMetrics",
    "aggregate_to_video_level",
    "evaluate_video_predictions",
    "evaluate_from_csv",
    "run_robustness_tests",
    "create_degraded_variants",
    "DEGRADATION_FUNCS",
]
