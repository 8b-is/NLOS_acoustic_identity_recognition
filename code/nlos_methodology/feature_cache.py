# -*- coding: utf-8 -*-
"""cache for STFT, LPVG, and fusion feature matrices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .config import CACHE_ROOT, CACHE_SCHEMA_VERSION, N_MICROPHONES
from .features import build_feature_matrix, segment_feature_dimension
from .metadata import manuscript_tset_label, public_tset_members, tset_storage_tag
from .preprocessing import (
    INDEPENDENT_WINDOW_STRATEGY,
    normalize_window_strategy,
    window_strategy_storage_tag,
)
from .utils import write_json


def feature_cache_directory(
    tset: Sequence[str],
    feature_name: str,
    window_strategy: str = INDEPENDENT_WINDOW_STRATEGY,
) -> Path:
    strategy = normalize_window_strategy(window_strategy)
    tset_root = CACHE_ROOT / f"TSET_{tset_storage_tag(tset)}"
    if strategy == INDEPENDENT_WINDOW_STRATEGY:
        return tset_root / "features" / feature_name
    return (
        tset_root
        / "windowing"
        / window_strategy_storage_tag(strategy)
        / "features"
        / feature_name
    )


def load_feature_cache(
    matrix_path: Path,
    metadata_path: Path,
    expected_samples: int,
    expected_dimension: int,
    tset: Sequence[str],
    feature_name: str,
    window_strategy: str = INDEPENDENT_WINDOW_STRATEGY,
) -> Optional[np.ndarray]:
    strategy = normalize_window_strategy(window_strategy)
    try:
        if not metadata_path.exists():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_metadata = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "tset_tag": tset_storage_tag(tset),
            "feature": feature_name,
            "n_samples": expected_samples,
            "feature_dimension": expected_dimension,
        }
        if strategy != INDEPENDENT_WINDOW_STRATEGY:
            expected_metadata["window_strategy"] = strategy

        for key, expected_value in expected_metadata.items():
            if metadata.get(key) != expected_value:
                print(
                    f"[CACHE] Ignored feature cache with mismatched {key}: "
                    f"{matrix_path}"
                )
                return None

        cached_strategy = metadata.get("window_strategy")
        if cached_strategy is not None:
            if normalize_window_strategy(str(cached_strategy)) != strategy:
                print(f"[CACHE] Ignored feature cache with wrong strategy: {matrix_path}")
                return None

        matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float32)
        if matrix.shape != (expected_samples, expected_dimension):
            print(
                f"[CACHE] Ignored incompatible feature cache {matrix_path}: "
                f"shape={matrix.shape}, expected="
                f"({expected_samples}, {expected_dimension})"
            )
            return None
        print(
            f"[CACHE] Loaded {window_strategy_storage_tag(strategy)} "
            f"features from: {matrix_path}"
        )
        return matrix
    except Exception as exc:
        print(f"[CACHE] Could not load feature cache {matrix_path}: {exc}")
        return None


def get_or_build_features(
    raw_segments: np.ndarray,
    tset: Sequence[str],
    feature_name: str,
    window_strategy: str = INDEPENDENT_WINDOW_STRATEGY,
) -> np.ndarray:
    strategy = normalize_window_strategy(window_strategy)
    expected_dimension = (
        N_MICROPHONES * len(tset) * segment_feature_dimension(feature_name)
    )
    directory = feature_cache_directory(
        tset,
        feature_name,
        window_strategy=strategy,
    )
    matrix_path = directory / "features.npy"
    metadata_path = directory / "metadata.json"

    if matrix_path.exists():
        loaded = load_feature_cache(
            matrix_path,
            metadata_path,
            expected_samples=len(raw_segments),
            expected_dimension=expected_dimension,
            tset=tset,
            feature_name=feature_name,
            window_strategy=strategy,
        )
        if loaded is not None:
            return loaded

    matrix = build_feature_matrix(raw_segments, feature_name)
    if matrix.shape != (len(raw_segments), expected_dimension):
        raise RuntimeError(
            f"Unexpected feature matrix shape {matrix.shape}; expected "
            f"({len(raw_segments)}, {expected_dimension})."
        )

    directory.mkdir(parents=True, exist_ok=True)
    np.save(matrix_path, matrix)
    write_json(
        metadata_path,
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "window_strategy": strategy,
            "tset_tag": tset_storage_tag(tset),
            "tset": manuscript_tset_label(tset),
            "tset_members": public_tset_members(tset),
            "feature": feature_name,
            "n_samples": int(len(raw_segments)),
            "feature_dimension": int(expected_dimension),
        },
    )
    print(
        f"[CACHE] Saved {window_strategy_storage_tag(strategy)} "
        f"features to: {matrix_path}"
    )
    return matrix
