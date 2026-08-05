# -*- coding: utf-8 -*-
"""
Two window-selection strategies are implemented:

``independent``
    The adaptive energy threshold and final above-threshold window are
    determined separately for every microphone channel. This is the main
    manuscript method.

``synchronized``
    Window energies at the same time index are averaged across the eight
    microphone channels. The adaptive threshold is calculated from this common
    energy sequence, and the final above-threshold window index is applied to
    every channel. This is the synchronized-window ablation method.
"""

from __future__ import annotations

from typing import Optional

import librosa
import numpy as np

from .config import (
    ENERGY_THRESHOLD_MULTIPLIER,
    N_MICROPHONES,
    PREEMPHASIS_COEFFICIENT,
    SECONDARY_WINDOW_SIZE,
)


INDEPENDENT_WINDOW_STRATEGY = "independent"
SYNCHRONIZED_WINDOW_STRATEGY = "synchronized"
WINDOW_STRATEGIES = (
    INDEPENDENT_WINDOW_STRATEGY,
    SYNCHRONIZED_WINDOW_STRATEGY,
)


def normalize_window_strategy(window_strategy: str) -> str:
    normalized = str(window_strategy).strip().lower()
    if normalized not in WINDOW_STRATEGIES:
        raise ValueError(
            f"Unknown window strategy: {window_strategy!r}. "
            f"Expected one of {WINDOW_STRATEGIES}."
        )
    return normalized


def window_strategy_label(window_strategy: str) -> str:
    normalized = normalize_window_strategy(window_strategy)
    if normalized == INDEPENDENT_WINDOW_STRATEGY:
        return "Independent window"
    return "Synchronized window"


def window_strategy_storage_tag(window_strategy: str) -> str:
    normalized = normalize_window_strategy(window_strategy)
    if normalized == INDEPENDENT_WINDOW_STRATEGY:
        return "Independent"
    return "Synchronized"


def preprocess_signal(signal: np.ndarray) -> np.ndarray:
    processed = np.asarray(signal)
    if np.iscomplexobj(processed):
        processed = np.real(processed)
    processed = processed.astype(np.float32, copy=False)
    processed = librosa.effects.preemphasis(
        processed,
        coef=PREEMPHASIS_COEFFICIENT,
    ).astype(np.float32, copy=False)

    peak = float(np.max(np.abs(processed))) if processed.size else 0.0
    if peak == 0.0:
        return processed
    return (processed / peak).astype(np.float32, copy=False)


def compute_nonoverlapping_window_energies(
    channel: np.ndarray,
    window_size: int = SECONDARY_WINDOW_SIZE,
) -> np.ndarray:
    values = np.asarray(channel, dtype=np.float32).reshape(-1)
    if values.size <= window_size:
        return np.empty(0, dtype=np.float64)

    starts = range(0, values.size - window_size, window_size)
    return np.asarray(
        [
            np.sum(np.abs(values[start : start + window_size] ** 2))
            for start in starts
        ],
        dtype=np.float64,
    )


def select_last_above_threshold_window(
    energies: np.ndarray,
    threshold_multiplier: float = ENERGY_THRESHOLD_MULTIPLIER,
) -> Optional[int]:
    energy_values = np.asarray(energies, dtype=np.float64).reshape(-1)
    if energy_values.size == 0:
        return None

    threshold = float(np.mean(energy_values)) * float(threshold_multiplier)
    above_threshold = np.flatnonzero(energy_values > threshold)
    if above_threshold.size == 0:
        return None
    return int(above_threshold[-1])


def extract_window_by_index(
    channel: np.ndarray,
    window_index: int,
    window_size: int = SECONDARY_WINDOW_SIZE,
) -> Optional[np.ndarray]:
    values = np.asarray(channel, dtype=np.float32).reshape(-1)
    start = int(window_index) * int(window_size)
    if start < 0 or start >= values.size:
        return None

    end = min(start + int(window_size), values.size)
    segment = values[start:end]
    if segment.size < window_size:
        segment = np.pad(segment, (0, window_size - segment.size))
    return segment.astype(np.float32, copy=False)


def detect_secondary_echo_window_index(
    channel: np.ndarray,
    window_size: int = SECONDARY_WINDOW_SIZE,
    threshold_multiplier: float = ENERGY_THRESHOLD_MULTIPLIER,
) -> Optional[int]:
    energies = compute_nonoverlapping_window_energies(channel, window_size)
    return select_last_above_threshold_window(energies, threshold_multiplier)


def detect_secondary_echo(
    channel: np.ndarray,
    window_size: int = SECONDARY_WINDOW_SIZE,
    threshold_multiplier: float = ENERGY_THRESHOLD_MULTIPLIER,
) -> Optional[np.ndarray]:
    selected_window = detect_secondary_echo_window_index(
        channel,
        window_size=window_size,
        threshold_multiplier=threshold_multiplier,
    )
    if selected_window is None:
        return None
    return extract_window_by_index(channel, selected_window, window_size)


def independent_secondary_echo_window_indices(
    waveform: np.ndarray,
    n_microphones: int = N_MICROPHONES,
    window_size: int = SECONDARY_WINDOW_SIZE,
    threshold_multiplier: float = ENERGY_THRESHOLD_MULTIPLIER,
) -> Optional[np.ndarray]:
    multichannel = np.asarray(waveform)
    if multichannel.ndim == 1 or multichannel.shape[0] < n_microphones:
        return None

    indices = []
    for channel_index in range(n_microphones):
        selected = detect_secondary_echo_window_index(
            multichannel[channel_index],
            window_size=window_size,
            threshold_multiplier=threshold_multiplier,
        )
        if selected is None:
            return None
        indices.append(selected)
    return np.asarray(indices, dtype=np.int64)


def synchronized_secondary_echo_window_index(
    waveform: np.ndarray,
    n_microphones: int = N_MICROPHONES,
    window_size: int = SECONDARY_WINDOW_SIZE,
    threshold_multiplier: float = ENERGY_THRESHOLD_MULTIPLIER,
) -> Optional[int]:
    multichannel = np.asarray(waveform)
    if multichannel.ndim == 1 or multichannel.shape[0] < n_microphones:
        return None

    channel_energies = [
        compute_nonoverlapping_window_energies(
            multichannel[channel_index],
            window_size=window_size,
        )
        for channel_index in range(n_microphones)
    ]
    if not channel_energies or any(item.size == 0 for item in channel_energies):
        return None

    common_length = min(item.size for item in channel_energies)
    if common_length == 0:
        return None

    energy_matrix = np.stack(
        [item[:common_length] for item in channel_energies],
        axis=0,
    )
    common_energy_sequence = np.mean(energy_matrix, axis=0)
    return select_last_above_threshold_window(
        common_energy_sequence,
        threshold_multiplier=threshold_multiplier,
    )


def extract_independent_secondary_echoes(
    waveform: np.ndarray,
    n_microphones: int = N_MICROPHONES,
    window_size: int = SECONDARY_WINDOW_SIZE,
    threshold_multiplier: float = ENERGY_THRESHOLD_MULTIPLIER,
) -> Optional[np.ndarray]:
    multichannel = np.asarray(waveform)
    indices = independent_secondary_echo_window_indices(
        multichannel,
        n_microphones=n_microphones,
        window_size=window_size,
        threshold_multiplier=threshold_multiplier,
    )
    if indices is None:
        return None

    segments = []
    for channel_index, selected_window in enumerate(indices):
        segment = extract_window_by_index(
            multichannel[channel_index],
            int(selected_window),
            window_size=window_size,
        )
        if segment is None:
            return None
        segments.append(segment)
    return np.stack(segments, axis=0).astype(np.float32, copy=False)


def extract_synchronized_secondary_echoes(
    waveform: np.ndarray,
    n_microphones: int = N_MICROPHONES,
    window_size: int = SECONDARY_WINDOW_SIZE,
    threshold_multiplier: float = ENERGY_THRESHOLD_MULTIPLIER,
) -> Optional[np.ndarray]:
    multichannel = np.asarray(waveform)
    selected_window = synchronized_secondary_echo_window_index(
        multichannel,
        n_microphones=n_microphones,
        window_size=window_size,
        threshold_multiplier=threshold_multiplier,
    )
    if selected_window is None:
        return None

    segments = []
    for channel_index in range(n_microphones):
        segment = extract_window_by_index(
            multichannel[channel_index],
            selected_window,
            window_size=window_size,
        )
        if segment is None:
            return None
        segments.append(segment)
    return np.stack(segments, axis=0).astype(np.float32, copy=False)


def extract_multichannel_secondary_echoes(
    waveform: np.ndarray,
    window_strategy: str = INDEPENDENT_WINDOW_STRATEGY,
    n_microphones: int = N_MICROPHONES,
) -> Optional[np.ndarray]:
    strategy = normalize_window_strategy(window_strategy)
    if strategy == INDEPENDENT_WINDOW_STRATEGY:
        return extract_independent_secondary_echoes(
            waveform,
            n_microphones=n_microphones,
        )
    return extract_synchronized_secondary_echoes(
        waveform,
        n_microphones=n_microphones,
    )
