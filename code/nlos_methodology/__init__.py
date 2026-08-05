# -*- coding: utf-8 -*-
"""NLOS identity-recognition methodology package."""

from .dataset import prepare_raw_segments
from .features import (
    compute_stft_magnitude,
    extract_lpvg_features,
    extract_sample_features,
    extract_segment_features,
    extract_stft_features,
)
from .human_empty_validation import (
    HumanEmptyConfig,
    RAW_AMPLITUDE_MODE,
    run_human_empty_validation,
)
from .metadata import (
    FEATURES,
    PUBLIC_SUBJECT_GENDER,
    PUBLIC_SUBJECT_ORDER,
    T_SETS,
)
from .preprocessing import (
    INDEPENDENT_WINDOW_STRATEGY,
    SYNCHRONIZED_WINDOW_STRATEGY,
    detect_secondary_echo,
    extract_independent_secondary_echoes,
    extract_synchronized_secondary_echoes,
    independent_secondary_echo_window_indices,
    preprocess_signal,
    synchronized_secondary_echo_window_index,
)
from .protocols import iter_object_configuration_loso
from .window_ablation import run_synchronized_window_ablation
from .visualization import (
    draw_confusion_matrix,
    draw_confusion_matrix_from_csv,
    load_confusion_matrix,
)

__all__ = [
    "HumanEmptyConfig",
    "RAW_AMPLITUDE_MODE",
    "FEATURES",
    "PUBLIC_SUBJECT_GENDER",
    "PUBLIC_SUBJECT_ORDER",
    "T_SETS",
    "INDEPENDENT_WINDOW_STRATEGY",
    "SYNCHRONIZED_WINDOW_STRATEGY",
    "compute_stft_magnitude",
    "detect_secondary_echo",
    "extract_independent_secondary_echoes",
    "extract_lpvg_features",
    "extract_sample_features",
    "extract_segment_features",
    "extract_stft_features",
    "extract_synchronized_secondary_echoes",
    "independent_secondary_echo_window_indices",
    "iter_object_configuration_loso",
    "load_confusion_matrix",
    "draw_confusion_matrix_from_csv",
    "draw_confusion_matrix",
    "prepare_raw_segments",
    "run_human_empty_validation",
    "run_synchronized_window_ablation",
    "preprocess_signal",
    "synchronized_secondary_echo_window_index",
]
