# -*- coding: utf-8 -*-
"""STFT, LPVG, fusion, and feature-matrix construction."""

from __future__ import annotations

import re

import librosa
import numpy as np
from joblib import Parallel, delayed

from .config import (
    LPVG_DEGREES,
    LPVG_TARGET_POINTS,
    N_JOBS_FEATURE,
    PARALLEL_BACKEND,
    PARALLEL_BATCH_SIZE,
    PARALLEL_VERBOSE,
    STFT_EXPECTED_BINS,
    STFT_HOP_LENGTH,
    STFT_N_FFT,
    STFT_WIN_LENGTH,
)
from .preprocessing import preprocess_signal


def compute_stft_magnitude(signal: np.ndarray) -> np.ndarray:
    processed = preprocess_signal(signal)
    stft = librosa.stft(
        y=processed,
        n_fft=STFT_N_FFT,
        hop_length=STFT_HOP_LENGTH,
        win_length=STFT_WIN_LENGTH,
        window="hann",
        center=True,
    )
    return np.abs(stft)


def extract_stft_features(signal: np.ndarray) -> np.ndarray:
    magnitude = compute_stft_magnitude(signal)
    if magnitude.size == 0:
        return np.zeros(STFT_EXPECTED_BINS, dtype=np.float32)

    time_averaged_magnitude = np.mean(magnitude, axis=1)
    features = np.log1p(time_averaged_magnitude)
    if features.size != STFT_EXPECTED_BINS:
        raise RuntimeError(
            f"Unexpected STFT feature length: {features.size}; "
            f"expected {STFT_EXPECTED_BINS}."
        )
    return features.astype(np.float32, copy=False)


def lpvg_degree_sequence(
    series: np.ndarray,
    penetrable_distance: int,
    target_points: int = LPVG_TARGET_POINTS,
) -> np.ndarray:
    values = np.asarray(series, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros(target_points, dtype=np.float32)

    source_positions = np.arange(values.size, dtype=np.float64)
    target_positions = np.linspace(0, values.size - 1, target_points)
    resampled = np.interp(target_positions, source_positions, values).astype(
        np.float32
    )

    n_points = resampled.size
    degrees = np.zeros(n_points, dtype=np.int32)

    for left in range(n_points - 1):
        left_value = float(resampled[left])
        for right in range(left + 1, n_points):
            right_value = float(resampled[right])
            blocked = 0
            for middle in range(left + 1, right):
                middle_value = float(resampled[middle])
                line_height = right_value + (left_value - right_value) * (
                    right - middle
                ) / (right - left)
                if middle_value > line_height:
                    blocked += 1
                    if blocked > penetrable_distance:
                        break
            if blocked <= penetrable_distance:
                degrees[left] += 1
                degrees[right] += 1

    return degrees.astype(np.float32, copy=False)


def extract_lpvg_features(
    signal: np.ndarray,
    penetrable_distance: int,
) -> np.ndarray:
    processed = preprocess_signal(signal)
    return lpvg_degree_sequence(
        processed,
        penetrable_distance=penetrable_distance,
        target_points=LPVG_TARGET_POINTS,
    )


def parse_lpvg_degree(feature_name: str) -> int:
    match = re.search(r"_D(\d+)$", feature_name)
    if match is None:
        raise ValueError(f"LPVG degree is missing from feature: {feature_name}")
    degree = int(match.group(1))
    if degree not in LPVG_DEGREES:
        raise ValueError(f"Unsupported LPVG degree: {degree}")
    return degree


def extract_segment_features(signal: np.ndarray, feature_name: str) -> np.ndarray:
    if feature_name == "STFT":
        return extract_stft_features(signal)

    if feature_name.startswith("LPVG_D"):
        return extract_lpvg_features(
            signal,
            penetrable_distance=parse_lpvg_degree(feature_name),
        )

    if feature_name.startswith("Fusion_D"):
        degree = parse_lpvg_degree(feature_name)
        return np.concatenate(
            [
                extract_stft_features(signal),
                extract_lpvg_features(signal, penetrable_distance=degree),
            ]
        ).astype(np.float32, copy=False)

    raise ValueError(f"Unknown feature representation: {feature_name}")


def extract_sample_features(
    sample_segments: np.ndarray,
    feature_name: str,
) -> np.ndarray:
    segment_features = [
        extract_segment_features(segment, feature_name)
        for segment in np.asarray(sample_segments)
    ]
    return np.stack(segment_features, axis=0).reshape(-1).astype(
        np.float32,
        copy=False,
    )


def build_feature_matrix(
    raw_segments: np.ndarray,
    feature_name: str,
) -> np.ndarray:
    if len(raw_segments) == 0:
        return np.empty((0, 0), dtype=np.float32)

    features = Parallel(
        n_jobs=N_JOBS_FEATURE,
        backend=PARALLEL_BACKEND,
        verbose=PARALLEL_VERBOSE,
        batch_size=PARALLEL_BATCH_SIZE,
    )(
        delayed(extract_sample_features)(raw_segments[index], feature_name)
        for index in range(len(raw_segments))
    )
    return np.stack(features, axis=0).astype(np.float32, copy=False)


def segment_feature_dimension(feature_name: str) -> int:
    if feature_name == "STFT":
        return STFT_EXPECTED_BINS
    if feature_name.startswith("LPVG_D"):
        parse_lpvg_degree(feature_name)
        return LPVG_TARGET_POINTS
    if feature_name.startswith("Fusion_D"):
        parse_lpvg_degree(feature_name)
        return STFT_EXPECTED_BINS + LPVG_TARGET_POINTS
    raise ValueError(f"Unknown feature representation: {feature_name}")
