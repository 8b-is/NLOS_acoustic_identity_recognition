# -*- coding: utf-8 -*-
"""WAV discovery, sample assembly, secondary-echo extraction, and raw cache."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np

from .config import (
    ALLOWED_SURFACES,
    CACHE_ROOT,
    CACHE_SCHEMA_VERSION,
    N_MICROPHONES,
    SAMPLE_RATE,
    SECONDARY_WINDOW_SIZE,
    VALID_POSTURES,
)
from .metadata import (
    PUBLIC_SUBJECT_GENDER,
    SUBJECTS_BY_SURFACE,
    public_tset_members,
    subject_genders,
    tset_storage_tag,
)
from .parsing import parse_audio_filename
from .preprocessing import (
    INDEPENDENT_WINDOW_STRATEGY,
    extract_multichannel_secondary_echoes,
    normalize_window_strategy,
    window_strategy_storage_tag,
)

RawDataTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def raw_cache_path(
    tset: Sequence[str],
    window_strategy: str = INDEPENDENT_WINDOW_STRATEGY,
) -> Path:

    strategy = normalize_window_strategy(window_strategy)
    tset_root = CACHE_ROOT / f"TSET_{tset_storage_tag(tset)}"
    if strategy == INDEPENDENT_WINDOW_STRATEGY:
        return tset_root / "raw_segments.npz"
    return (
        tset_root
        / "windowing"
        / window_strategy_storage_tag(strategy)
        / "raw_segments.npz"
    )


def validate_raw_cache(
    raw_segments: np.ndarray,
    subjects: np.ndarray,
    genders: np.ndarray,
    scenes: np.ndarray,
    surfaces: np.ndarray,
    expected_segments_per_sample: int,
) -> bool:
    n_samples = len(subjects)
    if not (
        len(raw_segments)
        == len(genders)
        == len(scenes)
        == len(surfaces)
        == n_samples
    ):
        return False

    if any(str(subject) not in SUBJECTS_BY_SURFACE["D"] for subject in subjects):
        return False

    expected_genders = subject_genders(subjects)
    if not np.array_equal(np.asarray(genders, dtype=object), expected_genders):
        return False

    if n_samples == 0:
        return True

    first = np.asarray(raw_segments[0])
    return (
        first.ndim == 2
        and first.shape[0] == expected_segments_per_sample
        and first.shape[1] == SECONDARY_WINDOW_SIZE
    )


def load_raw_cache(
    path: Path,
    expected_segments_per_sample: int,
    window_strategy: str = INDEPENDENT_WINDOW_STRATEGY,
) -> Optional[RawDataTuple]:
    strategy = normalize_window_strategy(window_strategy)
    try:
        with np.load(path, allow_pickle=False) as data:
            if "cache_schema_version" not in data.files:
                print(f"[CACHE] Ignored cache without release schema: {path}")
                return None
            cache_version = str(data["cache_schema_version"].item())
            if cache_version != CACHE_SCHEMA_VERSION:
                print(f"[CACHE] Ignored cache with schema {cache_version}: {path}")
                return None

            if "window_strategy" in data.files:
                cached_strategy = normalize_window_strategy(
                    str(data["window_strategy"].item())
                )
                if cached_strategy != strategy:
                    print(f"[CACHE] Ignored cache with wrong strategy: {path}")
                    return None
            elif strategy != INDEPENDENT_WINDOW_STRATEGY:
                print(f"[CACHE] Ignored cache without strategy metadata: {path}")
                return None

            raw_segments = np.asarray(data["raw_segments"], dtype=np.float32)
            subjects = np.asarray(data["subjects"], dtype=str)
            genders = np.asarray(data["genders"], dtype=str)
            scenes = np.asarray(data["scenes"], dtype=str)
            surfaces = np.asarray(data["surfaces"], dtype=str)

        if not validate_raw_cache(
            raw_segments,
            subjects,
            genders,
            scenes,
            surfaces,
            expected_segments_per_sample,
        ):
            print(f"[CACHE] Ignored incompatible raw cache: {path}")
            return None

        print(
            f"[CACHE] Loaded {window_strategy_storage_tag(strategy)} "
            f"raw segments from: {path}"
        )
        print(f"[CACHE] Raw-segment shape: {raw_segments.shape}")
        return raw_segments, subjects, genders, scenes, surfaces
    except Exception as exc:
        print(f"[CACHE] Could not load raw cache {path}: {exc}")
        return None


def save_raw_cache(
    path: Path,
    raw_segments: np.ndarray,
    subjects: np.ndarray,
    genders: np.ndarray,
    scenes: np.ndarray,
    surfaces: np.ndarray,
    tset: Sequence[str],
    window_strategy: str = INDEPENDENT_WINDOW_STRATEGY,
) -> None:
    strategy = normalize_window_strategy(window_strategy)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cache_schema_version=np.asarray(CACHE_SCHEMA_VERSION),
        window_strategy=np.asarray(strategy),
        tset_tag=np.asarray(tset_storage_tag(tset)),
        tset_members=np.asarray(public_tset_members(tset), dtype=str),
        raw_segments=raw_segments.astype(np.float32, copy=False),
        subjects=np.asarray(subjects, dtype=str),
        genders=np.asarray(genders, dtype=str),
        scenes=np.asarray(scenes, dtype=str),
        surfaces=np.asarray(surfaces, dtype=str),
    )
    print(
        f"[CACHE] Saved {window_strategy_storage_tag(strategy)} "
        f"raw segments to: {path}"
    )


def prepare_raw_segments(
    root_folder: Path,
    selected_tset: Sequence[str],
    window_strategy: str = INDEPENDENT_WINDOW_STRATEGY,
) -> RawDataTuple:
    """Prepare classification samples using the selected window strategy."""
    strategy = normalize_window_strategy(window_strategy)
    expected_segments = N_MICROPHONES * len(selected_tset)
    cache_path = raw_cache_path(selected_tset, strategy)

    if cache_path.exists():
        loaded = load_raw_cache(
            cache_path,
            expected_segments,
            window_strategy=strategy,
        )
        if loaded is not None:
            return loaded

    if not root_folder.exists():
        raise FileNotFoundError(
            f"Audio root does not exist: {root_folder}. Set NLOS_AUDIO_ROOT "
            "or edit ROOT_AUDIO_FOLDER."
        )

    groups: Dict[Tuple[str, str, str, str, str], Dict[str, Path]] = {}
    total_wav_files = 0

    for subdirectory, _directory_names, filenames in os.walk(root_folder):
        for filename in filenames:
            if not filename.lower().endswith(".wav"):
                continue

            total_wav_files += 1
            metadata = parse_audio_filename(filename)
            if metadata is None:
                continue
            if metadata["surface"] not in ALLOWED_SURFACES:
                continue
            if metadata["posture"] not in VALID_POSTURES:
                continue
            if metadata["excitation"] not in selected_tset:
                continue

            key = (
                metadata["surface"],
                metadata["scene"],
                metadata["subject"],
                metadata["posture"],
                metadata["direction"],
            )
            groups.setdefault(key, {})[metadata["excitation"]] = (
                Path(subdirectory) / filename
            )

    print(f"[DATA] Window strategy: {window_strategy_storage_tag(strategy)}")
    print(f"[DATA] WAV files scanned: {total_wav_files}")
    print(f"[DATA] Candidate measurement groups: {len(groups)}")

    raw_samples: List[np.ndarray] = []
    subjects: List[str] = []
    genders: List[str] = []
    scenes: List[str] = []
    surfaces: List[str] = []

    for key, excitation_paths in groups.items():
        surface, scene, subject, _posture, _direction = key
        if not all(excitation in excitation_paths for excitation in selected_tset):
            continue

        sample_segments: List[np.ndarray] = []
        valid_group = True

        for excitation in selected_tset:
            wav_path = excitation_paths[excitation]
            try:
                waveform, _ = librosa.load(
                    wav_path,
                    sr=SAMPLE_RATE,
                    mono=False,
                )
            except Exception as exc:
                print(f"[DATA] Skipped unreadable file {wav_path}: {exc}")
                valid_group = False
                break

            waveform = np.asarray(waveform)
            recording_segments = extract_multichannel_secondary_echoes(
                waveform,
                window_strategy=strategy,
                n_microphones=N_MICROPHONES,
            )
            if recording_segments is None:
                print(
                    f"[DATA] No secondary-echo segment: {wav_path}, "
                    f"strategy={window_strategy_storage_tag(strategy)}"
                )
                valid_group = False
                break

            # Preserve excitation-major, microphone-minor ordering.
            sample_segments.extend(recording_segments)

        if not valid_group:
            continue

        sample_array = np.stack(sample_segments, axis=0).astype(
            np.float32,
            copy=False,
        )
        if sample_array.shape != (expected_segments, SECONDARY_WINDOW_SIZE):
            raise RuntimeError(
                f"Unexpected raw sample shape {sample_array.shape}; expected "
                f"({expected_segments}, {SECONDARY_WINDOW_SIZE})."
            )

        raw_samples.append(sample_array)
        subjects.append(subject)
        genders.append(PUBLIC_SUBJECT_GENDER[subject])
        scenes.append(scene)
        surfaces.append(surface)

    if raw_samples:
        raw_segments = np.stack(raw_samples, axis=0).astype(np.float32, copy=False)
    else:
        raw_segments = np.empty(
            (0, expected_segments, SECONDARY_WINDOW_SIZE),
            dtype=np.float32,
        )

    subject_array = np.asarray(subjects, dtype=str)
    gender_array = np.asarray(genders, dtype=str)
    scene_array = np.asarray(scenes, dtype=str)
    surface_array = np.asarray(surfaces, dtype=str)

    print(f"[DATA] Valid classification samples: {len(raw_segments)}")
    print(f"[DATA] Raw-segment shape: {raw_segments.shape}")

    save_raw_cache(
        cache_path,
        raw_segments,
        subject_array,
        gender_array,
        scene_array,
        surface_array,
        selected_tset,
        window_strategy=strategy,
    )
    return raw_segments, subject_array, gender_array, scene_array, surface_array
