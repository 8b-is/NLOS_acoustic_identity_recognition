# -*- coding: utf-8 -*-
"""

This module implements the seven PyTorch models used in the manuscript:

- 1D-CNN on complete raw secondary-echo windows
- 2D-CNN on stream-averaged STFT spectrograms
- 2D-CNN-MultiStream on stream-preserved STFT spectrograms
- LSTM and BiLSTM on complete raw secondary-echo windows
- LSTM-STFT and BiLSTM-STFT on the exact classical STFT vectors

"""

from __future__ import annotations

import gc
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from . import dataset as dataset_module
from .config import OUTPUT_ROOT, ROOT_AUDIO_FOLDER, ensure_output_directories
from .feature_cache import get_or_build_features
from .metadata import PUBLIC_SUBJECT_ORDER, tset_storage_tag
from .config import GLOBAL_SEED

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as torch_import_error:  
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_IMPORT_ERROR: Optional[Exception] = torch_import_error
else:
    TORCH_IMPORT_ERROR = None


# Fixed methodology

TSET: Tuple[str, ...] = (
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "C"
)
WINDOW_STRATEGY = "independent"

ALL_DEEP_MODELS: Tuple[str, ...] = (
    "1D-CNN",
    "2D-CNN",
    "2D-CNN-MultiStream",
    "LSTM",
    "BiLSTM",
    "LSTM-STFT",
    "BiLSTM-STFT",
)

PREEMPHASIS_COEFFICIENT = 0.97
NORMALIZATION_EPSILON = 1e-12

STFT_N_FFT = 512
STFT_WIN_LENGTH = 512
STFT_HOP_LENGTH = 256
STFT_WINDOW = "hann"
STFT_CENTER = True
STFT_LOG_EPSILON = 1e-8
STFT_FREQUENCY_BINS = STFT_N_FFT // 2 + 1  # 257, including DC and Nyquist
STFT_EXPECTED_FRAMES = 5

RAW_WINDOW_SAMPLES = 1024
EXPECTED_STREAMS = 72  # 9 excitation conditions x 8 microphones
N_CLASSES = 10

CSV_SEPARATOR = ";"

RESULT_DIRECTORY = (
    OUTPUT_ROOT
    / "results"
    / "Deep_Learning"
    / f"TSET_{tset_storage_tag(TSET)}"
)


@dataclass(frozen=True)
class DeepLearningConfig:

    models: Tuple[str, ...] = ALL_DEEP_MODELS
    epochs: int = 60
    patience: int = 10
    batch_size: int = 16
    validation_fraction: float = 0.15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    num_workers: int = 0
    save_model_weights: bool = False
    force_rebuild_tensors: bool = False
    device: Optional[str] = None



# Output utilities
def _save_csv(
    dataframe: pd.DataFrame,
    path: Path,
    *,
    index: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_csv(
        temporary_path,
        index=index,
        encoding="utf-8-sig",
        sep=CSV_SEPARATOR,
        lineterminator="\n",
    )
    temporary_path.replace(path)


def _save_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


# Reproducibility
def _seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# Shared modular-pipeline access
def _load_raw_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load raw selected segments, public subject labels, and scene labels."""

    if hasattr(dataset_module, "prepare_raw_dataset"):
        bundle = dataset_module.prepare_raw_dataset(
            ROOT_AUDIO_FOLDER,
            TSET,
            window_strategy=WINDOW_STRATEGY,
        )
        return (
            np.asarray(bundle.raw_segments, dtype=object),
            np.asarray(bundle.subjects, dtype=str),
            np.asarray(bundle.scenes, dtype=str),
        )

    try:
        output = dataset_module.prepare_raw_segments(
            ROOT_AUDIO_FOLDER,
            TSET,
            window_strategy=WINDOW_STRATEGY,
        )
    except TypeError:
        output = dataset_module.prepare_raw_segments(
            ROOT_AUDIO_FOLDER,
            TSET,
        )

    if len(output) < 4:
        raise ValueError(
            "prepare_raw_segments() must return raw segments, subject labels, "
            "and scene labels."
        )

    return (
        np.asarray(output[0], dtype=object),
        np.asarray(output[1], dtype=str),
        np.asarray(output[3], dtype=str),
    )


def _load_exact_stft_features(raw_segments: np.ndarray) -> np.ndarray:
    """Load the exact classical time-averaged log-STFT feature matrix."""

    try:
        matrix = get_or_build_features(
            raw_segments,
            TSET,
            "STFT",
            window_strategy=WINDOW_STRATEGY,
        )
    except TypeError:
        matrix = get_or_build_features(
            raw_segments,
            TSET,
            "STFT",
        )

    matrix = np.asarray(matrix, dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError(
            f"The classical STFT feature matrix must be 2D, got {matrix.shape}."
        )

    if matrix.shape[0] != len(raw_segments):
        raise ValueError(
            "The classical STFT feature/sample counts are not aligned: "
            f"{matrix.shape[0]} != {len(raw_segments)}."
        )

    if not np.isfinite(matrix).all():
        raise ValueError("The classical STFT feature matrix contains NaN or Inf.")

    return matrix


# Raw-sample conversion and deterministic preprocessing
def _sample_to_stream_matrix(sample: Any) -> np.ndarray:
    """Convert a nested raw sample to [streams, samples]."""

    streams: List[np.ndarray] = []

    def visit(value: Any) -> None:
        if value is None:
            return

        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            array = np.asarray(value, dtype=object)

        if array.dtype != object and array.ndim == 1:
            stream = np.real(array).astype(np.float32, copy=False)
            if stream.size:
                streams.append(stream)
            return

        if array.dtype != object and array.ndim >= 2:
            reshaped = array.reshape(-1, array.shape[-1])
            for row in reshaped:
                stream = np.real(row).astype(np.float32, copy=False)
                if stream.size:
                    streams.append(stream)
            return

        if array.dtype == object:
            try:
                numeric = np.asarray(array, dtype=np.float32)
            except (TypeError, ValueError):
                numeric = None

            if numeric is not None:
                if numeric.ndim == 1:
                    if numeric.size:
                        streams.append(np.real(numeric).astype(np.float32, copy=False))
                    return
                if numeric.ndim >= 2:
                    for row in numeric.reshape(-1, numeric.shape[-1]):
                        if row.size:
                            streams.append(np.real(row).astype(np.float32, copy=False))
                    return

            for item in array.reshape(-1).tolist():
                visit(item)
            return

        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(sample)

    if not streams:
        return np.zeros((0, 0), dtype=np.float32)

    minimum_length = min(len(stream) for stream in streams)
    if minimum_length <= 0:
        return np.zeros((0, 0), dtype=np.float32)

    return np.stack(
        [stream[:minimum_length] for stream in streams],
        axis=0,
    ).astype(np.float32, copy=False)


def _preprocess_stream(signal: np.ndarray) -> np.ndarray:
    """Apply pre-emphasis 0.97 and per-stream peak normalization."""

    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    if np.iscomplexobj(values):
        values = np.real(values).astype(np.float32, copy=False)

    if values.size == 0:
        return values

    processed = librosa.effects.preemphasis(
        values,
        coef=PREEMPHASIS_COEFFICIENT,
    ).astype(np.float32, copy=False)

    peak = float(np.max(np.abs(processed)))
    if peak <= NORMALIZATION_EPSILON:
        return processed

    return (processed / peak).astype(np.float32, copy=False)


def _stft_magnitude(signal: np.ndarray) -> np.ndarray:
    """Return the complete 257-bin one-sided STFT magnitude matrix."""

    processed = _preprocess_stream(signal)
    spectrum = librosa.stft(
        y=processed,
        n_fft=STFT_N_FFT,
        hop_length=STFT_HOP_LENGTH,
        win_length=STFT_WIN_LENGTH,
        window=STFT_WINDOW,
        center=STFT_CENTER,
    )
    magnitude = np.abs(spectrum).astype(np.float32, copy=False)

    if magnitude.shape[0] != STFT_FREQUENCY_BINS:
        raise ValueError(
            "Unexpected STFT frequency-bin count: "
            f"{magnitude.shape[0]} != {STFT_FREQUENCY_BINS}."
        )

    return magnitude


# Tensor construction
def build_raw_tensor(raw_segments: np.ndarray) -> np.ndarray:
    """Return [samples, 1024 time points, 72 streams]."""

    tensors: List[np.ndarray] = []

    for sample_index, sample in enumerate(raw_segments):
        streams = _sample_to_stream_matrix(sample)

        if streams.shape != (EXPECTED_STREAMS, RAW_WINDOW_SAMPLES):
            raise ValueError(
                f"Sample {sample_index} has raw shape {streams.shape}; expected "
                f"({EXPECTED_STREAMS}, {RAW_WINDOW_SAMPLES})."
            )

        processed = np.stack(
            [_preprocess_stream(stream) for stream in streams],
            axis=0,
        )
        tensors.append(processed.T.astype(np.float32, copy=False))

    if not tensors:
        return np.zeros(
            (0, RAW_WINDOW_SAMPLES, EXPECTED_STREAMS),
            dtype=np.float32,
        )

    return np.stack(tensors, axis=0).astype(np.float32, copy=False)


def build_stream_averaged_stft_tensor(
    raw_segments: np.ndarray,
) -> np.ndarray:
    """Return [samples, 257 frequency bins, 5 frames, 1]."""

    tensors: List[np.ndarray] = []

    for sample_index, sample in enumerate(raw_segments):
        streams = _sample_to_stream_matrix(sample)

        if streams.shape != (EXPECTED_STREAMS, RAW_WINDOW_SAMPLES):
            raise ValueError(
                f"Sample {sample_index} has raw shape {streams.shape}; expected "
                f"({EXPECTED_STREAMS}, {RAW_WINDOW_SAMPLES})."
            )

        matrices = [_stft_magnitude(stream) for stream in streams]
        frame_counts = {matrix.shape[1] for matrix in matrices}

        if len(frame_counts) != 1:
            raise ValueError(
                f"Sample {sample_index} contains inconsistent STFT frame counts: "
                f"{sorted(frame_counts)}."
            )

        frame_count = next(iter(frame_counts))
        if frame_count != STFT_EXPECTED_FRAMES:
            raise ValueError(
                f"Sample {sample_index} has {frame_count} STFT frames; expected "
                f"{STFT_EXPECTED_FRAMES}."
            )

        stack = np.stack(matrices, axis=0)
        image = np.log(
            np.mean(stack, axis=0) + STFT_LOG_EPSILON
        )[..., None]
        tensors.append(image.astype(np.float32, copy=False))

    if not tensors:
        return np.zeros(
            (0, STFT_FREQUENCY_BINS, STFT_EXPECTED_FRAMES, 1),
            dtype=np.float32,
        )

    return np.stack(tensors, axis=0).astype(np.float32, copy=False)


def build_multistream_stft_tensor(
    raw_segments: np.ndarray,
) -> np.ndarray:
    """Return [samples, 257 frequency bins, 5 frames, 72 streams]."""

    tensors: List[np.ndarray] = []

    for sample_index, sample in enumerate(raw_segments):
        streams = _sample_to_stream_matrix(sample)

        if streams.shape != (EXPECTED_STREAMS, RAW_WINDOW_SAMPLES):
            raise ValueError(
                f"Sample {sample_index} has raw shape {streams.shape}; expected "
                f"({EXPECTED_STREAMS}, {RAW_WINDOW_SAMPLES})."
            )

        matrices = [_stft_magnitude(stream) for stream in streams]
        frame_counts = {matrix.shape[1] for matrix in matrices}

        if len(frame_counts) != 1:
            raise ValueError(
                f"Sample {sample_index} contains inconsistent STFT frame counts: "
                f"{sorted(frame_counts)}."
            )

        frame_count = next(iter(frame_counts))
        if frame_count != STFT_EXPECTED_FRAMES:
            raise ValueError(
                f"Sample {sample_index} has {frame_count} STFT frames; expected "
                f"{STFT_EXPECTED_FRAMES}."
            )

        stack = np.stack(matrices, axis=-1)
        image = np.log(stack + STFT_LOG_EPSILON)
        tensors.append(image.astype(np.float32, copy=False))

    if not tensors:
        return np.zeros(
            (
                0,
                STFT_FREQUENCY_BINS,
                STFT_EXPECTED_FRAMES,
                EXPECTED_STREAMS,
            ),
            dtype=np.float32,
        )

    return np.stack(tensors, axis=0).astype(np.float32, copy=False)


def build_stft_vector_sequence_tensor(
    stft_feature_matrix: np.ndarray,
) -> np.ndarray:
    """Return exact classical STFT vectors as [samples, 257, 72]."""

    matrix = np.asarray(stft_feature_matrix, dtype=np.float32)
    expected_dimension = EXPECTED_STREAMS * STFT_FREQUENCY_BINS

    if matrix.ndim != 2:
        raise ValueError(f"STFT feature matrix must be 2D, got {matrix.shape}.")

    if matrix.shape[1] != expected_dimension:
        raise ValueError(
            "The exact STFT feature dimension is incompatible with the manuscript "
            f"layout: {matrix.shape[1]} != {EXPECTED_STREAMS} x "
            f"{STFT_FREQUENCY_BINS} ({expected_dimension})."
        )

    return (
        matrix.reshape(
            matrix.shape[0],
            EXPECTED_STREAMS,
            STFT_FREQUENCY_BINS,
        )
        .transpose(0, 2, 1)
        .astype(np.float32, copy=False)
    )


# PyTorch architectures
if nn is not None:

    class RawEcho1DCNN(nn.Module):
        def __init__(self, n_classes: int = N_CLASSES) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(EXPECTED_STREAMS, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.30),
                nn.Linear(128, n_classes),
            )

        def forward(self, values: Any) -> Any:
            values = values.permute(0, 2, 1)
            return self.classifier(self.features(values))


    class STFT2DCNN(nn.Module):
        def __init__(self, n_classes: int = N_CLASSES) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 24, kernel_size=(7, 3), padding=(3, 1)),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),
                nn.Conv2d(24, 48, kernel_size=(5, 3), padding=(2, 1)),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),
                nn.Conv2d(48, 96, kernel_size=3, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.30),
                nn.Linear(96, n_classes),
            )

        def forward(self, values: Any) -> Any:
            values = values.permute(0, 3, 1, 2)
            return self.classifier(self.features(values))


    class MultiStreamSTFT2DCNN(nn.Module):
        def __init__(self, n_classes: int = N_CLASSES) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(
                    EXPECTED_STREAMS,
                    32,
                    kernel_size=(7, 3),
                    padding=(3, 1),
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),
                nn.Conv2d(32, 64, kernel_size=(5, 3), padding=(2, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),
                nn.Conv2d(64, 96, kernel_size=3, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.30),
                nn.Linear(96, n_classes),
            )

        def forward(self, values: Any) -> Any:
            values = values.permute(0, 3, 1, 2)
            return self.classifier(self.features(values))


    class SequenceLSTM(nn.Module):
        def __init__(
            self,
            input_features: int = EXPECTED_STREAMS,
            n_classes: int = N_CLASSES,
        ) -> None:
            super().__init__()
            projection_size = 64
            hidden_size = 64
            self.input_norm = nn.LayerNorm(input_features)
            self.input_projection = nn.Sequential(
                nn.Linear(input_features, projection_size),
                nn.ReLU(inplace=True),
            )
            self.lstm = nn.LSTM(
                input_size=projection_size,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=0.30,
                bidirectional=False,
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.30),
                nn.Linear(hidden_size * 2, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.20),
                nn.Linear(128, n_classes),
            )

        def forward(self, values: Any) -> Any:
            values = self.input_norm(values)
            values = self.input_projection(values)
            sequence, _ = self.lstm(values)
            mean_pool = torch.mean(sequence, dim=1)
            max_pool = torch.amax(sequence, dim=1)
            return self.classifier(
                torch.cat([mean_pool, max_pool], dim=1)
            )


    class SequenceBiLSTM(nn.Module):
        def __init__(
            self,
            input_features: int = EXPECTED_STREAMS,
            n_classes: int = N_CLASSES,
        ) -> None:
            super().__init__()
            projection_size = 64
            hidden_size = 64
            self.input_norm = nn.LayerNorm(input_features)
            self.input_projection = nn.Sequential(
                nn.Linear(input_features, projection_size),
                nn.ReLU(inplace=True),
            )
            self.lstm = nn.LSTM(
                input_size=projection_size,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=0.30,
                bidirectional=True,
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.30),
                nn.Linear(hidden_size * 4, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.20),
                nn.Linear(128, n_classes),
            )

        def forward(self, values: Any) -> Any:
            values = self.input_norm(values)
            values = self.input_projection(values)
            sequence, _ = self.lstm(values)
            mean_pool = torch.mean(sequence, dim=1)
            max_pool = torch.amax(sequence, dim=1)
            return self.classifier(
                torch.cat([mean_pool, max_pool], dim=1)
            )


# Dataset validation and model registry
def _natural_scene_key(value: str) -> Tuple[int, str]:
    text = str(value)
    try:
        return int(text), text
    except ValueError:
        return 10**9, text


def _to_public_subject_label(value: str) -> str:
    mapping = {
        "C1": "S1",
        "C2": "S2",
        "C3": "S3",
        "C4": "S4",
        "C6": "S5",
        "C7": "S6",
        "C8": "S7",
        "C9": "S8",
        "C10": "S9",
        "C11": "S10",
    }
    return mapping.get(str(value), str(value))


def _validate_dataset(
    raw_segments: np.ndarray,
    subjects: np.ndarray,
    scenes: np.ndarray,
    stft_features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not (
        len(raw_segments)
        == len(subjects)
        == len(scenes)
        == len(stft_features)
    ):
        raise ValueError(
            "Raw segments, labels, scenes, and STFT features are not aligned."
        )

    public_subjects = np.asarray(
        [_to_public_subject_label(value) for value in subjects],
        dtype=str,
    )
    expected_subjects = set(PUBLIC_SUBJECT_ORDER)
    observed_subjects = set(public_subjects.tolist())

    if observed_subjects != expected_subjects:
        raise ValueError(
            "The deep-learning analysis requires exactly S1-S10. "
            f"Observed: {sorted(observed_subjects)}."
        )

    unique_scenes = sorted(
        set(scenes.tolist()),
        key=_natural_scene_key,
    )
    if len(unique_scenes) != 10:
        raise ValueError(
            "The deep-learning protocol requires 10 object-"
            f"configuration LOSO folds; observed {len(unique_scenes)} scenes."
        )

    subject_to_index = {
        subject: index
        for index, subject in enumerate(PUBLIC_SUBJECT_ORDER)
    }
    encoded_labels = np.asarray(
        [subject_to_index[subject] for subject in public_subjects],
        dtype=np.int64,
    )

    # Validate the raw shape once before expensive tensor construction.
    for sample_index, sample in enumerate(raw_segments):
        shape = _sample_to_stream_matrix(sample).shape
        if shape != (EXPECTED_STREAMS, RAW_WINDOW_SAMPLES):
            raise ValueError(
                f"Sample {sample_index} has shape {shape}; expected "
                f"({EXPECTED_STREAMS}, {RAW_WINDOW_SAMPLES})."
            )

    expected_feature_dimension = EXPECTED_STREAMS * STFT_FREQUENCY_BINS
    if stft_features.shape[1] != expected_feature_dimension:
        raise ValueError(
            "The STFT-vector recurrent inputs require "
            f"{expected_feature_dimension} values per sample, but the shared "
            f"STFT matrix has {stft_features.shape[1]}."
        )

    return public_subjects, encoded_labels, unique_scenes


def _model_registry() -> Dict[str, Dict[str, Any]]:
    if nn is None:
        return {}

    return {
        "1D-CNN": {
            "architecture_id": "RawEcho_1D_CNN_PyTorch",
            "input_representation": "Raw secondary echo",
            "input_dimensions": "1024 × 72",
            "tensor_key": "raw",
            "factory": lambda: RawEcho1DCNN(),
        },
        "2D-CNN": {
            "architecture_id": "STFT_2D_CNN_PyTorch",
            "input_representation": "Stream-averaged STFT spectrogram",
            "input_dimensions": "257 × 5 × 1",
            "tensor_key": "stft_averaged",
            "factory": lambda: STFT2DCNN(),
        },
        "2D-CNN-MultiStream": {
            "architecture_id": "STFT_MultiStream_2D_CNN_PyTorch",
            "input_representation": (
                "STFT spectrogram with streams preserved as separate channels"
            ),
            "input_dimensions": "257 × 5 × 72",
            "tensor_key": "stft_multistream",
            "factory": lambda: MultiStreamSTFT2DCNN(),
        },
        "LSTM": {
            "architecture_id": "RawEcho_LSTM_PyTorch",
            "input_representation": "Raw secondary echo",
            "input_dimensions": "1024 × 72",
            "tensor_key": "raw",
            "factory": lambda: SequenceLSTM(),
        },
        "BiLSTM": {
            "architecture_id": "RawEcho_BiLSTM_PyTorch",
            "input_representation": "Raw secondary echo",
            "input_dimensions": "1024 × 72",
            "tensor_key": "raw",
            "factory": lambda: SequenceBiLSTM(),
        },
        "LSTM-STFT": {
            "architecture_id": "ExactSTFTVector_LSTM_PyTorch",
            "input_representation": "Time-averaged log-STFT vector",
            "input_dimensions": "257 × 72",
            "tensor_key": "stft_vector",
            "factory": lambda: SequenceLSTM(),
        },
        "BiLSTM-STFT": {
            "architecture_id": "ExactSTFTVector_BiLSTM_PyTorch",
            "input_representation": "Time-averaged log-STFT vector",
            "input_dimensions": "257 × 72",
            "tensor_key": "stft_vector",
            "factory": lambda: SequenceBiLSTM(),
        },
    }


# Fold-only standardization and validation split
def _stratified_validation_indices(
    labels: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    training_indices: List[int] = []
    validation_indices: List[int] = []

    for class_index in np.unique(labels):
        indices = np.flatnonzero(labels == class_index)
        rng.shuffle(indices)

        if len(indices) <= 1:
            training_indices.extend(indices.tolist())
            continue

        validation_count = max(
            1,
            int(round(len(indices) * validation_fraction)),
        )
        validation_count = min(validation_count, len(indices) - 1)

        validation_indices.extend(indices[:validation_count].tolist())
        training_indices.extend(indices[validation_count:].tolist())

    return (
        np.asarray(training_indices, dtype=np.int64),
        np.asarray(validation_indices, dtype=np.int64),
    )


def _fit_standardizer(
    training_tensor: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    axes = tuple(range(training_tensor.ndim - 1))
    mean = np.mean(
        training_tensor,
        axis=axes,
        keepdims=True,
        dtype=np.float64,
    ).astype(np.float32)
    standard_deviation = np.std(
        training_tensor,
        axis=axes,
        keepdims=True,
        dtype=np.float64,
    ).astype(np.float32)
    standard_deviation = np.where(
        standard_deviation < 1e-6,
        1.0,
        standard_deviation,
    ).astype(np.float32)
    return mean, standard_deviation


def _apply_standardizer(
    tensor: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
) -> np.ndarray:
    return (
        (tensor.astype(np.float32, copy=False) - mean)
        / standard_deviation
    ).astype(np.float32, copy=False)


def _class_weight_tensor(
    labels: np.ndarray,
    device: Any,
) -> Any:
    counts = np.bincount(labels, minlength=N_CLASSES).astype(np.float64)
    weights = np.zeros(N_CLASSES, dtype=np.float32)
    valid = counts > 0
    weights[valid] = len(labels) / (N_CLASSES * counts[valid])
    return torch.tensor(weights, dtype=torch.float32, device=device)


# Tensor cache
def _load_or_build_tensor(
    tensor_key: str,
    raw_segments: np.ndarray,
    stft_features: np.ndarray,
    cache_directory: Path,
    force_rebuild: bool,
) -> np.ndarray:
    cache_directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "raw": cache_directory / "raw_1024x72.npy",
        "stft_averaged": cache_directory / "stft_257x5x1.npy",
        "stft_multistream": cache_directory / "stft_257x5x72.npy",
        "stft_vector": cache_directory / "stft_vector_257x72.npy",
    }

    expected_shapes = {
        "raw": (RAW_WINDOW_SAMPLES, EXPECTED_STREAMS),
        "stft_averaged": (STFT_FREQUENCY_BINS, STFT_EXPECTED_FRAMES, 1),
        "stft_multistream": (
            STFT_FREQUENCY_BINS,
            STFT_EXPECTED_FRAMES,
            EXPECTED_STREAMS,
        ),
        "stft_vector": (STFT_FREQUENCY_BINS, EXPECTED_STREAMS),
    }

    builders: Dict[str, Callable[[], np.ndarray]] = {
        "raw": lambda: build_raw_tensor(raw_segments),
        "stft_averaged": lambda: build_stream_averaged_stft_tensor(raw_segments),
        "stft_multistream": lambda: build_multistream_stft_tensor(raw_segments),
        "stft_vector": lambda: build_stft_vector_sequence_tensor(stft_features),
    }

    if tensor_key not in paths:
        raise KeyError(f"Unknown tensor key: {tensor_key}")

    path = paths[tensor_key]
    expected_sample_shape = expected_shapes[tensor_key]

    if path.exists() and not force_rebuild:
        tensor = np.load(path).astype(np.float32, copy=False)
    else:
        tensor = builders[tensor_key]().astype(np.float32, copy=False)
        np.save(path, tensor)

    if tensor.shape[0] != len(raw_segments):
        raise ValueError(
            f"Cached tensor {path.name} has {tensor.shape[0]} samples; expected "
            f"{len(raw_segments)}. Delete the stale cache or enable "
            "force_rebuild_tensors."
        )

    if tuple(tensor.shape[1:]) != expected_sample_shape:
        raise ValueError(
            f"Cached tensor {path.name} has shape {tensor.shape[1:]}; expected "
            f"{expected_sample_shape}. Delete the stale cache or enable "
            "force_rebuild_tensors."
        )

    if not np.isfinite(tensor).all():
        raise ValueError(f"Tensor {path.name} contains NaN or Inf.")

    return tensor


# Training and evaluation
def _classification_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def _evaluate_model(
    model: Any,
    loader: Any,
    device: Any,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    true_batches: List[np.ndarray] = []
    prediction_batches: List[np.ndarray] = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch_values, batch_labels in loader:
            batch_values = batch_values.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            logits = model(batch_values)
            loss = criterion(logits, batch_labels)

            total_loss += float(loss.item()) * len(batch_labels)
            total_count += len(batch_labels)
            true_batches.append(batch_labels.cpu().numpy())
            prediction_batches.append(
                torch.argmax(logits, dim=1).cpu().numpy()
            )

    true_values = (
        np.concatenate(true_batches)
        if true_batches
        else np.empty(0, dtype=np.int64)
    )
    predictions = (
        np.concatenate(prediction_batches)
        if prediction_batches
        else np.empty(0, dtype=np.int64)
    )

    return (
        true_values,
        predictions,
        total_loss / max(total_count, 1),
    )


def _resolve_device(config: DeepLearningConfig) -> Any:
    if torch is None:
        return None

    if config.device:
        requested = torch.device(config.device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device requested ({config.device}) but CUDA is unavailable."
            )
        return requested

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_one_fold(
    model_name: str,
    model_specification: Mapping[str, Any],
    tensor: np.ndarray,
    encoded_labels: np.ndarray,
    public_subjects: np.ndarray,
    scenes: np.ndarray,
    held_out_scene: str,
    fold_number: int,
    config: DeepLearningConfig,
    device: Any,
    result_directory: Path,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    _seed_everything(GLOBAL_SEED + fold_number)

    outer_test_mask = scenes == held_out_scene
    outer_train_mask = ~outer_test_mask

    if not np.any(outer_train_mask) or not np.any(outer_test_mask):
        raise ValueError(
            f"Invalid LOSO split for held-out scene {held_out_scene}."
        )

    outer_training_tensor = tensor[outer_train_mask]
    outer_training_labels = encoded_labels[outer_train_mask]
    test_tensor_unscaled = tensor[outer_test_mask]
    test_labels = encoded_labels[outer_test_mask]
    test_subjects = public_subjects[outer_test_mask]

    local_training_indices, local_validation_indices = (
        _stratified_validation_indices(
            outer_training_labels,
            config.validation_fraction,
            GLOBAL_SEED + fold_number,
        )
    )

    if (
        len(local_training_indices) == 0
        or len(local_validation_indices) == 0
    ):
        raise RuntimeError(
            f"Inner validation split failed for {model_name}, scene "
            f"{held_out_scene}."
        )

    mean, standard_deviation = _fit_standardizer(
        outer_training_tensor[local_training_indices]
    )

    training_tensor = _apply_standardizer(
        outer_training_tensor[local_training_indices],
        mean,
        standard_deviation,
    )
    validation_tensor = _apply_standardizer(
        outer_training_tensor[local_validation_indices],
        mean,
        standard_deviation,
    )
    test_tensor = _apply_standardizer(
        test_tensor_unscaled,
        mean,
        standard_deviation,
    )

    training_labels = outer_training_labels[local_training_indices]
    validation_labels = outer_training_labels[local_validation_indices]

    training_dataset = TensorDataset(
        torch.from_numpy(training_tensor),
        torch.from_numpy(training_labels.astype(np.int64)),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(validation_tensor),
        torch.from_numpy(validation_labels.astype(np.int64)),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_tensor),
        torch.from_numpy(test_labels.astype(np.int64)),
    )

    generator = torch.Generator()
    generator.manual_seed(GLOBAL_SEED + fold_number)
    pin_memory = device.type == "cuda"

    training_loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    model = model_specification["factory"]().to(device)
    training_criterion = nn.CrossEntropyLoss(
        weight=_class_weight_tensor(training_labels, device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, config.patience // 3),
        min_lr=1e-6,
    )

    best_state: Optional[Dict[str, Any]] = None
    best_validation_loss = float("inf")
    best_validation_accuracy = float("nan")
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows: List[Dict[str, Any]] = []

    training_start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        training_loss_sum = 0.0
        training_count = 0
        training_correct = 0

        for batch_values, batch_labels in training_loader:
            batch_values = batch_values.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_values)
            loss = training_criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.gradient_clip_norm,
            )
            optimizer.step()

            training_loss_sum += float(loss.item()) * len(batch_labels)
            training_count += len(batch_labels)
            training_correct += int(
                (torch.argmax(logits, dim=1) == batch_labels).sum().item()
            )

        validation_true, validation_predictions, validation_loss = (
            _evaluate_model(model, validation_loader, device)
        )
        validation_accuracy = float(
            accuracy_score(validation_true, validation_predictions)
        )
        training_loss = training_loss_sum / max(training_count, 1)
        training_accuracy = training_correct / max(training_count, 1)

        scheduler.step(validation_loss)
        current_learning_rate = float(optimizer.param_groups[0]["lr"])

        history_rows.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "training_accuracy": training_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "learning_rate": current_learning_rate,
            }
        )

        if validation_loss < best_validation_loss - 1e-6:
            best_validation_loss = float(validation_loss)
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError(
            f"No checkpoint was produced for {model_name}, scene "
            f"{held_out_scene}."
        )

    model.load_state_dict(best_state)
    model.to(device)
    training_seconds = time.perf_counter() - training_start

    prediction_start = time.perf_counter()
    true_indices, predicted_indices, test_loss = _evaluate_model(
        model,
        test_loader,
        device,
    )
    prediction_seconds = time.perf_counter() - prediction_start

    metrics = _classification_metrics(true_indices, predicted_indices)
    parameter_count = int(
        sum(parameter.numel() for parameter in model.parameters())
    )

    fold_result = {
        "model": model_name,
        "architecture_id": model_specification["architecture_id"],
        "input_representation": model_specification["input_representation"],
        "input_dimensions": model_specification["input_dimensions"],
        "fold": fold_number,
        "held_out_scene": str(held_out_scene),
        "outer_train_n": int(np.sum(outer_train_mask)),
        "inner_train_n": int(len(training_tensor)),
        "validation_n": int(len(validation_tensor)),
        "test_n": int(len(test_tensor)),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "best_validation_accuracy": float(best_validation_accuracy),
        "test_loss": float(test_loss),
        "training_seconds": float(training_seconds),
        "prediction_seconds": float(prediction_seconds),
        "inference_ms_per_sample": (
            1000.0 * prediction_seconds / max(1, len(test_labels))
        ),
        "parameters": parameter_count,
        "device": str(device),
        **metrics,
    }

    history = pd.DataFrame(history_rows)
    predictions = pd.DataFrame(
        {
            "model": model_name,
            "fold": fold_number,
            "held_out_scene": str(held_out_scene),
            "true_subject": test_subjects,
            "predicted_subject": np.asarray(
                [PUBLIC_SUBJECT_ORDER[index] for index in predicted_indices],
                dtype=object,
            ),
            "true_class_id": true_indices,
            "predicted_class_id": predicted_indices,
            "correct": true_indices == predicted_indices,
        }
    )

    if config.save_model_weights:
        weights_directory = (
            result_directory
            / "model_weights"
            / str(model_specification["architecture_id"])
        )
        weights_directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model_name,
                "architecture_id": model_specification["architecture_id"],
                "held_out_scene": str(held_out_scene),
                "state_dict": best_state,
                "normalization_mean": mean,
                "normalization_standard_deviation": standard_deviation,
                "public_subject_order": list(PUBLIC_SUBJECT_ORDER),
                "configuration": asdict(config),
            },
            weights_directory / f"scene_{held_out_scene}.pt",
        )

    del (
        model,
        optimizer,
        scheduler,
        training_loader,
        validation_loader,
        test_loader,
        training_dataset,
        validation_dataset,
        test_dataset,
        training_tensor,
        validation_tensor,
        test_tensor,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fold_result, history, predictions


# Summary table
def _build_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    registry = _model_registry()
    rows: List[Dict[str, Any]] = []

    for model_name in ALL_DEEP_MODELS:
        group = fold_metrics[fold_metrics["model"] == model_name]
        if group.empty:
            continue

        mean_accuracy_percent = float(group["accuracy"].mean() * 100.0)
        standard_deviation_percent = float(
            group["accuracy"].std(ddof=1) * 100.0
        )

        rows.append(
            {
                "Model": model_name,
                "Input Representation": registry[model_name][
                    "input_representation"
                ],
                "Input Dimensions": registry[model_name]["input_dimensions"],
                "Mean ACC (%)": mean_accuracy_percent,
                "SD (%)": standard_deviation_percent,
                "Mean ACC ± SD (%)": (
                    f"{mean_accuracy_percent:.2f} ± "
                    f"{standard_deviation_percent:.2f}"
                ),
                "Mean Macro Precision (%)": float(
                    group["macro_precision"].mean() * 100.0
                ),
                "Mean Macro Recall (%)": float(
                    group["macro_recall"].mean() * 100.0
                ),
                "Mean Macro F1 (%)": float(
                    group["macro_f1"].mean() * 100.0
                ),
                "Mean Best Epoch": float(group["best_epoch"].mean()),
                "Mean Training Seconds": float(
                    group["training_seconds"].mean()
                ),
                "Mean Inference ms/sample": float(
                    group["inference_ms_per_sample"].mean()
                ),
                "Parameters": int(group["parameters"].iloc[0]),
                "N LOSO Folds": int(len(group)),
            }
        )

    return pd.DataFrame(rows)


# Public entry point
def run_deep_learning_baselines(
    config: DeepLearningConfig = DeepLearningConfig(),
) -> Dict[str, Path]:

    ensure_output_directories()
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    selected_models = tuple(config.models)
    unknown_models = sorted(set(selected_models) - set(ALL_DEEP_MODELS))
    if unknown_models:
        raise ValueError(
            f"Unknown deep-learning model(s): {unknown_models}. "
            f"Allowed: {list(ALL_DEEP_MODELS)}."
        )

    if torch is None or nn is None:
        status_path = RESULT_DIRECTORY / "deep_learning_status.csv"
        _save_csv(
            pd.DataFrame(
                [
                    {
                        "status": "NOT_RUN",
                        "reason": repr(TORCH_IMPORT_ERROR),
                        "required_action": (
                            "Install a compatible PyTorch build. No surrogate "
                            "model was substituted."
                        ),
                    }
                ]
            ),
            status_path,
        )
        return {"status": status_path}

    _seed_everything(GLOBAL_SEED)
    device = _resolve_device(config)

    raw_segments, subjects, scenes = _load_raw_data()
    stft_features = _load_exact_stft_features(raw_segments)
    public_subjects, encoded_labels, unique_scenes = _validate_dataset(
        raw_segments,
        subjects,
        scenes,
        stft_features,
    )

    configuration_path = RESULT_DIRECTORY / "deep_model_configuration.json"
    _save_json(
        {
            "tset": list(TSET),
            "window_strategy": WINDOW_STRATEGY,
            "models": list(selected_models),
            "global_seed": GLOBAL_SEED,
            "outer_validation": (
                "10-fold object-configuration Leave-One-Scene-Out"
            ),
            "inner_validation_fraction": config.validation_fraction,
            "inner_validation": (
                "Class-stratified split from the current outer training fold only"
            ),
            "normalization": (
                "Fit on the inner-training subset only; then applied to inner "
                "validation and outer test data"
            ),
            "optimizer": "AdamW",
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "maximum_epochs": config.epochs,
            "early_stopping_patience": config.patience,
            "gradient_clip_norm": config.gradient_clip_norm,
            "scheduler": (
                "ReduceLROnPlateau(mode=min, factor=0.5, "
                f"patience={max(2, config.patience // 3)}, min_lr=1e-6)"
            ),
            "preemphasis_coefficient": PREEMPHASIS_COEFFICIENT,
            "segment_normalization": "Per-stream peak normalization",
            "raw_input": "1024 x 72",
            "stft": {
                "n_fft": STFT_N_FFT,
                "win_length": STFT_WIN_LENGTH,
                "hop_length": STFT_HOP_LENGTH,
                "window": STFT_WINDOW,
                "center": STFT_CENTER,
                "frequency_bins": STFT_FREQUENCY_BINS,
                "frames": STFT_EXPECTED_FRAMES,
                "dc_and_nyquist": "retained",
            },
            "device": str(device),
            "torch_version": str(torch.__version__),
            "configuration": asdict(config),
        },
        configuration_path,
    )

    registry = _model_registry()
    tensor_cache_directory = RESULT_DIRECTORY / "tensor_cache"
    fold_rows: List[Dict[str, Any]] = []
    prediction_tables: List[pd.DataFrame] = []
    current_tensor_key: Optional[str] = None
    current_tensor: Optional[np.ndarray] = None

    fold_metrics_path = RESULT_DIRECTORY / "deep_baseline_fold_metrics.csv"
    predictions_path = RESULT_DIRECTORY / "deep_baseline_predictions.csv"

    for model_name in selected_models:
        model_specification = registry[model_name]
        tensor_key = str(model_specification["tensor_key"])

        # Keep only one large representation in memory. A previously generated
        # representation is reloaded from its tensor cache when needed again.
        if current_tensor_key != tensor_key or current_tensor is None:
            if current_tensor is not None:
                del current_tensor
                gc.collect()
            current_tensor = _load_or_build_tensor(
                tensor_key,
                raw_segments,
                stft_features,
                tensor_cache_directory,
                config.force_rebuild_tensors,
            )
            current_tensor_key = tensor_key

        tensor = current_tensor
        print(
            f"[DEEP] Model={model_name} | input={tensor.shape[1:]} | "
            f"device={device}"
        )

        for fold_number, held_out_scene in enumerate(
            unique_scenes,
            start=1,
        ):
            print(
                f"[DEEP] {model_name} | fold {fold_number}/"
                f"{len(unique_scenes)} | held-out scene={held_out_scene}"
            )

            fold_result, history, predictions = _train_one_fold(
                model_name,
                model_specification,
                tensor,
                encoded_labels,
                public_subjects,
                scenes,
                held_out_scene,
                fold_number,
                config,
                device,
                RESULT_DIRECTORY,
            )
            fold_rows.append(fold_result)
            prediction_tables.append(predictions)

            history_path = (
                RESULT_DIRECTORY
                / "training_history"
                / str(model_specification["architecture_id"])
                / f"scene_{held_out_scene}.csv"
            )
            _save_csv(history, history_path)

            # Persist partial progress after every completed fold.
            _save_csv(pd.DataFrame(fold_rows), fold_metrics_path)
            _save_csv(
                pd.concat(prediction_tables, ignore_index=True),
                predictions_path,
            )

            print(
                f"[DEEP] {model_name} | scene={held_out_scene} | "
                f"accuracy={fold_result['accuracy'] * 100.0:.2f}% | "
                f"best_epoch={fold_result['best_epoch']}"
            )

    if current_tensor is not None:
        del current_tensor
        gc.collect()

    fold_metrics = pd.DataFrame(fold_rows)
    summary = _build_summary(fold_metrics)
    summary_path = RESULT_DIRECTORY / "Table_deep_learning_summary.csv"
    _save_csv(summary, summary_path)

    completion_path = RESULT_DIRECTORY / "RUN_COMPLETE.txt"
    completion_path.write_text(
        "Deep-learning baseline analysis completed.\n"
        f"Models: {', '.join(selected_models)}\n"
        f"LOSO folds per model: {len(unique_scenes)}\n"
        f"Total trained models: {len(selected_models) * len(unique_scenes)}\n"
        "All metrics were calculated from model predictions.\n",
        encoding="utf-8",
    )

    print(f"T-set               : {'+'.join('M' if x == 'C' else x for x in TSET)}")
    print(f"Deep models         : {len(selected_models)}")
    print(f"LOSO folds/model    : {len(unique_scenes)}")
    print(
        "Total model fits    : "
        f"{len(selected_models) * len(unique_scenes)}"
    )
    print("Manual result values: none")

    return {
        "fold_metrics": fold_metrics_path,
        "predictions": predictions_path,
        "table_1": summary_path,
        "configuration": configuration_path,
        "completion": completion_path,
    }
