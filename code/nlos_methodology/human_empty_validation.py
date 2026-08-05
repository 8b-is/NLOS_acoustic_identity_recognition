# -*- coding: utf-8 -*-
"""Matched subject-present versus subject-absent validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.io import wavfile

from .config import (
    ENERGY_THRESHOLD_MULTIPLIER,
    FIGURES_ROOT,
    N_MICROPHONES,
    RESULTS_ROOT,
    ROOT_AUDIO_FOLDER,
    SAMPLE_RATE,
    SECONDARY_WINDOW_SIZE,
    VALID_POSTURES,
)
from .metadata import PUBLIC_SUBJECT_ORDER, SCENE_DESCRIPTIONS
from .parsing import parse_audio_filename
from .preprocessing import (
    compute_nonoverlapping_window_energies,
    extract_window_by_index,
    select_last_above_threshold_window,
)
from .utils import atomic_write_text, write_json

CSV_SEPARATOR = ";"
EPSILON = np.finfo(np.float64).tiny
RAW_AMPLITUDE_MODE = "raw_no_normalization"
DEFAULT_REQUIRED_REPEATS = ("0", "1", "2")


@dataclass(frozen=True)
class HumanEmptyRecord:

    path: Path
    surface: str
    scene: str
    subject: str
    acquisition_subject: str
    human_present: bool
    posture: str
    direction: str
    transmitter: str
    repeat: Optional[str]


@dataclass
class HumanEmptyConfig:

    audio_root: Path
    results_dir: Path
    figures_dir: Path
    surface: str = "D"
    scenes: Tuple[str, ...] = tuple(str(index) for index in range(10))
    postures: Tuple[str, ...] = ("A", "O", "C")
    directions: Optional[Tuple[str, ...]] = None
    transmitters: Tuple[str, ...] = (
        "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "C"
    )
    pairing_mode: str = "same-session"
    empty_reference_session: Optional[str] = None
    required_repeats: Tuple[str, ...] = DEFAULT_REQUIRED_REPEATS
    sample_rate: int = SAMPLE_RATE
    window_size: int = SECONDARY_WINDOW_SIZE
    threshold_multiplier: float = ENERGY_THRESHOLD_MULTIPLIER
    max_channels: int = N_MICROPHONES
    max_human_files: Optional[int] = None
    bootstrap_iterations: int = 5000
    random_seed: int = 2026
    n_case_figures: int = 10
    case_selection_mode: str = "subject_balanced"
    case_pair_ids: Optional[Tuple[str, ...]] = None
    case_subjects: Optional[Tuple[str, ...]] = None
    case_scenes: Optional[Tuple[str, ...]] = None
    case_transmitters: Optional[Tuple[str, ...]] = None
    case_channels: Optional[Tuple[int, ...]] = None
    representative_channel: Optional[int] = None
    plot_only: bool = False
    metrics_dir: Optional[Path] = None
    amplitude_processing: str = RAW_AMPLITUDE_MODE

    def validate(self) -> None:
        if self.amplitude_processing != RAW_AMPLITUDE_MODE:
            raise ValueError("Only raw, non-normalized amplitude processing is allowed.")
        if self.pairing_mode not in {"same-session", "shared-session"}:
            raise ValueError("pairing_mode must be 'same-session' or 'shared-session'.")
        if self.pairing_mode == "shared-session" and not self.empty_reference_session:
            raise ValueError(
                "empty_reference_session is required for shared-session pairing."
            )
        if len(self.required_repeats) < 2:
            raise ValueError("At least two subject-absent repeats are required.")
        if self.max_channels < 1:
            raise ValueError("max_channels must be positive.")
        if self.window_size < 1:
            raise ValueError("window_size must be positive.")
        if self.bootstrap_iterations < 1:
            raise ValueError("bootstrap_iterations must be positive.")



# CSV and configuration helpers
def _write_csv(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_csv(
        temporary,
        sep=CSV_SEPARATOR,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    temporary.replace(path)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=CSV_SEPARATOR, encoding="utf-8-sig")


def _config_payload(config: HumanEmptyConfig) -> Dict[str, object]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, tuple):
            payload[key] = list(value)
    return payload


def _split_tokens(text: Optional[str], uppercase: bool = True) -> Optional[Tuple[str, ...]]:
    if text is None:
        return None
    cleaned = str(text).replace("|", ",").replace(";", ",")
    values = [item.strip() for item in cleaned.split(",") if item.strip()]
    if not values or any(item.upper() == "ALL" for item in values):
        return None
    if uppercase:
        values = [item.upper() for item in values]
    return tuple(values)


def _split_channels(text: Optional[str]) -> Optional[Tuple[int, ...]]:
    tokens = _split_tokens(text, uppercase=False)
    if tokens is None:
        return None
    channels: List[int] = []
    for token in tokens:
        value = int(token)
        if value < 1:
            raise ValueError("Case channels are one-based and must be positive.")
        channels.append(value)
    return tuple(channels)


def _subject_sort_key(label: str) -> Tuple[int, str]:
    text = str(label)
    try:
        return (PUBLIC_SUBJECT_ORDER.index(text), text)
    except ValueError:
        suffix = text[1:] if text.upper().startswith("S") else ""
        return (int(suffix) + len(PUBLIC_SUBJECT_ORDER) if suffix.isdigit() else 10**9, text)


# WAV reading and filename interpretation
def read_wav_raw(path: Path, target_sample_rate: int) -> np.ndarray:
    """Read WAV data as float64 without amplitude normalization or scaling."""
    sample_rate, data = wavfile.read(path)
    values = np.asarray(data)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"Unexpected WAV shape {values.shape}: {path}")

    # Casting preserves numeric sample values. No integer-range division is used.
    values = values.astype(np.float64, copy=False)
    if sample_rate != target_sample_rate:
        divisor = math.gcd(int(sample_rate), int(target_sample_rate))
        values = signal.resample_poly(
            values,
            target_sample_rate // divisor,
            sample_rate // divisor,
            axis=0,
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"NaN or Inf values in WAV file: {path}")
    return values


def parse_human_empty_record(path: Path) -> Optional[HumanEmptyRecord]:
    """Extend the shared filename parser with empty-repeat information."""
    metadata = parse_audio_filename(path.name)
    if metadata is None:
        return None

    parts = path.stem.split("_")
    if len(parts) < 7:
        return None
    acquisition_subject = parts[3].upper()
    field4 = parts[4].upper()

    if field4 == "Y":
        transmitter = parts[5].upper()
        repeat = parts[6]
        if transmitter == "BG":
            return None
        return HumanEmptyRecord(
            path=path,
            surface=str(metadata["surface"]).upper(),
            scene=str(metadata["scene"]),
            subject=str(metadata["subject"]),
            acquisition_subject=acquisition_subject,
            human_present=False,
            posture="NA",
            direction="NA",
            transmitter=transmitter,
            repeat=repeat,
        )

    if field4 in VALID_POSTURES:
        transmitter = str(metadata["excitation"]).upper()
        if transmitter == "BG":
            return None
        return HumanEmptyRecord(
            path=path,
            surface=str(metadata["surface"]).upper(),
            scene=str(metadata["scene"]),
            subject=str(metadata["subject"]),
            acquisition_subject=acquisition_subject,
            human_present=True,
            posture=str(metadata["posture"]).upper(),
            direction=str(metadata["direction"]).upper(),
            transmitter=transmitter,
            repeat=None,
        )
    return None


# Pairing
def build_matched_manifest(
    config: HumanEmptyConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pair each subject-present recording with matched active empty repeats."""
    config.validate()
    root = config.audio_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    scene_filter = set(config.scenes)
    posture_filter = set(config.postures)
    direction_filter = set(config.directions) if config.directions else None
    transmitter_filter = set(config.transmitters)

    humans: List[HumanEmptyRecord] = []
    empties: Dict[
        Tuple[str, str, str, str], Dict[str, List[HumanEmptyRecord]]
    ] = {}

    for path in sorted(root.rglob("*.wav")):
        record = parse_human_empty_record(path)
        if record is None or record.surface != config.surface.upper():
            continue
        if record.scene not in scene_filter:
            continue
        if record.transmitter not in transmitter_filter:
            continue
        if record.human_present:
            if record.posture not in posture_filter:
                continue
            if direction_filter is not None and record.direction not in direction_filter:
                continue
            humans.append(record)
        else:
            key = (
                record.surface,
                record.scene,
                record.acquisition_subject,
                record.transmitter,
            )
            empties.setdefault(key, {}).setdefault(str(record.repeat), []).append(record)

    humans.sort(
        key=lambda record: (
            _subject_sort_key(record.subject),
            int(record.scene) if record.scene.isdigit() else 10**9,
            record.transmitter,
            record.posture,
            record.direction,
            str(record.path),
        )
    )
    if config.max_human_files is not None:
        humans = humans[: max(0, int(config.max_human_files))]

    manifest_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    shared_session = (config.empty_reference_session or "").upper()

    for human in humans:
        empty_session = (
            human.acquisition_subject
            if config.pairing_mode == "same-session"
            else shared_session
        )
        key = (
            human.surface,
            human.scene,
            empty_session,
            human.transmitter,
        )
        by_repeat = empties.get(key, {})
        missing = [
            repeat
            for repeat in config.required_repeats
            if len(by_repeat.get(repeat, [])) == 0
        ]
        duplicate = [
            repeat
            for repeat in config.required_repeats
            if len(by_repeat.get(repeat, [])) > 1
        ]
        status = (
            "paired"
            if not missing and not duplicate
            else ("missing_repeat" if missing else "duplicate_repeat")
        )
        relative_name = str(human.path.relative_to(root))
        digest = hashlib.sha1(relative_name.encode("utf-8")).hexdigest()[:10]
        pair_id = (
            f"{human.subject}_{human.surface}_{human.scene}_"
            f"{human.posture}_{human.direction}_{human.transmitter}_{digest}"
        )
        selected_empty_paths = [
            by_repeat[repeat][0].path
            for repeat in config.required_repeats
            if len(by_repeat.get(repeat, [])) == 1
        ]

        audit_rows.append(
            {
                "pair_id": pair_id,
                "status": status,
                "pairing_mode": config.pairing_mode,
                "subject": human.subject,
                "human_session": human.acquisition_subject,
                "empty_session": empty_session,
                "surface": human.surface,
                "scene": human.scene,
                "transmitter": human.transmitter,
                "human_wav": str(human.path),
                "required_repeats": "|".join(config.required_repeats),
                "missing_repeats": "|".join(missing),
                "duplicate_repeats": "|".join(duplicate),
                "selected_empty_wavs": "|".join(
                    str(item) for item in selected_empty_paths
                ),
            }
        )
        if status != "paired":
            continue

        manifest_rows.append(
            {
                "pair_id": pair_id,
                "subject": human.subject,
                "session_subject": human.acquisition_subject,
                "empty_session_subject": empty_session,
                "surface": human.surface,
                "scene": human.scene,
                "scene_label": SCENE_DESCRIPTIONS.get(human.scene, human.scene),
                "posture": human.posture,
                "direction": human.direction,
                "transmitter": human.transmitter,
                "human_wav": str(human.path),
                "empty_wavs": "|".join(str(item) for item in selected_empty_paths),
                "n_empty_repeats": len(selected_empty_paths),
            }
        )

    return pd.DataFrame(manifest_rows), pd.DataFrame(audit_rows)


# Shared-window raw-amplitude analysis
def _window_selection(
    channel: np.ndarray,
    window_size: int,
    threshold_multiplier: float,
) -> Tuple[int, int, int, float, np.ndarray]:
    energies = compute_nonoverlapping_window_energies(channel, window_size)
    selected_index = select_last_above_threshold_window(
        energies,
        threshold_multiplier=threshold_multiplier,
    )
    if selected_index is None:
        raise ValueError("No window exceeds the adaptive threshold.")
    threshold = float(np.mean(energies)) * float(threshold_multiplier)
    start = int(selected_index) * int(window_size)
    return selected_index, start, start + window_size, threshold, energies


def _extract_same_window(
    channel: np.ndarray,
    window_index: int,
    window_size: int,
) -> np.ndarray:
    segment = extract_window_by_index(channel, window_index, window_size)
    if segment is None:
        return np.zeros(window_size, dtype=np.float64)
    return np.asarray(segment, dtype=np.float64)


def _energy(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sum(array * array))


def _db_ratio(numerator: float, denominator: float) -> float:
    return float(
        10.0
        * np.log10(
            (float(numerator) + EPSILON) / (float(denominator) + EPSILON)
        )
    )


def process_matched_manifest(
    manifest: pd.DataFrame,
    config: HumanEmptyConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []

    for row_index, record in manifest.iterrows():
        if row_index == 0 or (row_index + 1) % 50 == 0 or row_index + 1 == len(manifest):
            print(f"[HUMAN-EMPTY] {row_index + 1:,}/{len(manifest):,}")
        pair_id = str(record["pair_id"])
        try:
            human = read_wav_raw(Path(str(record["human_wav"])), config.sample_rate)
            empty_paths = [
                Path(item)
                for item in str(record["empty_wavs"]).split("|")
                if item
            ]
            empties = [read_wav_raw(path, config.sample_rate) for path in empty_paths]
            channel_count = min(
                config.max_channels,
                human.shape[1],
                *(empty.shape[1] for empty in empties),
            )

            for channel_index in range(channel_count):
                try:
                    human_channel = human[:, channel_index]
                    (
                        selected_index,
                        start,
                        stop,
                        threshold,
                        human_energy_curve,
                    ) = _window_selection(
                        human_channel,
                        config.window_size,
                        config.threshold_multiplier,
                    )

                    human_segment = _extract_same_window(
                        human_channel,
                        selected_index,
                        config.window_size,
                    )
                    empty_segments = [
                        _extract_same_window(
                            empty[:, channel_index],
                            selected_index,
                            config.window_size,
                        )
                        for empty in empties
                    ]
                    empty_mean_segment = np.mean(
                        np.stack(empty_segments, axis=0), axis=0
                    )
                    human_difference_energy = _energy(
                        human_segment - empty_mean_segment
                    )

                    leave_one_out_energies: List[float] = []
                    for repeat_index, candidate in enumerate(empty_segments):
                        other_segments = [
                            item
                            for index, item in enumerate(empty_segments)
                            if index != repeat_index
                        ]
                        other_mean = np.mean(np.stack(other_segments, axis=0), axis=0)
                        leave_one_out_energies.append(_energy(candidate - other_mean))

                    empty_control_median = float(np.median(leave_one_out_energies))
                    empty_control_maximum = float(np.max(leave_one_out_energies))
                    empty_energy_curves = [
                        compute_nonoverlapping_window_energies(
                            empty[:, channel_index], config.window_size
                        )
                        for empty in empties
                    ]
                    common_curve_length = min(
                        len(curve) for curve in empty_energy_curves
                    )
                    empty_mean_energy_curve = np.mean(
                        np.stack(
                            [
                                curve[:common_curve_length]
                                for curve in empty_energy_curves
                            ],
                            axis=0,
                        ),
                        axis=0,
                    )

                    rows.append(
                        {
                            "pair_id": pair_id,
                            "subject": record["subject"],
                            "session_subject": record["session_subject"],
                            "empty_session_subject": record["empty_session_subject"],
                            "surface": record["surface"],
                            "scene": record["scene"],
                            "scene_label": record["scene_label"],
                            "posture": record["posture"],
                            "direction": record["direction"],
                            "transmitter": record["transmitter"],
                            "channel": channel_index + 1,
                            "human_wav": record["human_wav"],
                            "empty_wavs": record["empty_wavs"],
                            "amplitude_processing": RAW_AMPLITUDE_MODE,
                            "selected_window_index": selected_index,
                            "selected_start_sample": start,
                            "selected_end_sample": stop,
                            "selected_start_ms": start * 1000.0 / config.sample_rate,
                            "selected_end_ms": stop * 1000.0 / config.sample_rate,
                            "adaptive_threshold": threshold,
                            "human_selected_raw_energy": float(
                                human_energy_curve[selected_index]
                            ),
                            "human_minus_empty_energy": human_difference_energy,
                            "empty_repeat_variation_median_energy": empty_control_median,
                            "empty_repeat_variation_max_energy": empty_control_maximum,
                            "contrast_vs_empty_median_db": _db_ratio(
                                human_difference_energy, empty_control_median
                            ),
                            "contrast_vs_empty_max_db": _db_ratio(
                                human_difference_energy, empty_control_maximum
                            ),
                            "human_exceeds_empty_control_max": bool(
                                human_difference_energy > empty_control_maximum
                            ),
                            "human_energy_curve_json": json.dumps(
                                human_energy_curve.tolist()
                            ),
                            "empty_mean_energy_curve_json": json.dumps(
                                empty_mean_energy_curve.tolist()
                            ),
                        }
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "pair_id": pair_id,
                            "channel": channel_index + 1,
                            "error": repr(exc),
                        }
                    )
        except Exception as exc:
            errors.append({"pair_id": pair_id, "channel": "", "error": repr(exc)})

    return (
        pd.DataFrame(rows),
        pd.DataFrame(errors, columns=["pair_id", "channel", "error"]),
    )


def aggregate_human_empty_metrics(
    channel_metrics: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate channel-level results to recording and subject levels."""
    record_rows: List[Dict[str, object]] = []
    for pair_id, group in channel_metrics.groupby("pair_id", sort=False):
        first = group.iloc[0]
        record_rows.append(
            {
                "pair_id": pair_id,
                "subject": first["subject"],
                "session_subject": first["session_subject"],
                "scene": first["scene"],
                "scene_label": first["scene_label"],
                "posture": first["posture"],
                "direction": first["direction"],
                "transmitter": first["transmitter"],
                "n_channels": len(group),
                "amplitude_processing": RAW_AMPLITUDE_MODE,
                "median_selected_start_ms": float(
                    group["selected_start_ms"].median()
                ),
                "selected_start_range_ms": float(
                    group["selected_start_ms"].max()
                    - group["selected_start_ms"].min()
                ),
                "median_contrast_db": float(
                    group["contrast_vs_empty_median_db"].median()
                ),
                "median_contrast_vs_max_db": float(
                    group["contrast_vs_empty_max_db"].median()
                ),
                "proportion_channels_exceeding_empty_max": float(
                    group["human_exceeds_empty_control_max"].mean()
                ),
                "human_wav": first["human_wav"],
                "empty_wavs": first["empty_wavs"],
            }
        )
    record_metrics = pd.DataFrame(record_rows)

    subject_rows: List[Dict[str, object]] = []
    for subject, group in record_metrics.groupby("subject", sort=False):
        subject_channels = channel_metrics[channel_metrics["subject"] == subject]
        subject_rows.append(
            {
                "subject": subject,
                "n_records": len(group),
                "n_channel_comparisons": len(subject_channels),
                "amplitude_processing": RAW_AMPLITUDE_MODE,
                "median_contrast_db": float(group["median_contrast_db"].median()),
                "median_contrast_vs_max_db": float(
                    group["median_contrast_vs_max_db"].median()
                ),
                "proportion_channels_exceeding_empty_max": float(
                    subject_channels["human_exceeds_empty_control_max"].mean()
                ),
                "median_selected_start_ms": float(
                    group["median_selected_start_ms"].median()
                ),
                "median_interchannel_range_ms": float(
                    group["selected_start_range_ms"].median()
                ),
            }
        )
    subject_metrics = pd.DataFrame(subject_rows)
    if not subject_metrics.empty:
        subject_metrics = subject_metrics.sort_values(
            "subject", key=lambda series: series.map(_subject_sort_key)
        ).reset_index(drop=True)
    return record_metrics, subject_metrics


def bootstrap_median_confidence_interval(
    values: Sequence[float],
    iterations: int,
    seed: int,
) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return float("nan"), float("nan")
    random_generator = np.random.default_rng(seed)
    bootstrap_values = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        bootstrap_values[index] = np.median(
            random_generator.choice(array, size=array.size, replace=True)
        )
    lower, upper = np.quantile(bootstrap_values, [0.025, 0.975])
    return float(lower), float(upper)


def make_validation_summary(
    channel_metrics: pd.DataFrame,
    record_metrics: pd.DataFrame,
    subject_metrics: pd.DataFrame,
    config: HumanEmptyConfig,
) -> pd.DataFrame:
    values = subject_metrics["median_contrast_db"].to_numpy(dtype=float)
    confidence_low, confidence_high = bootstrap_median_confidence_interval(
        values,
        iterations=config.bootstrap_iterations,
        seed=config.random_seed,
    )
    nonzero = values[np.isfinite(values) & (values != 0)]
    positive_count = int(np.sum(nonzero > 0))
    sign_test_p = (
        float(stats.binomtest(positive_count, len(nonzero), 0.5).pvalue)
        if len(nonzero)
        else float("nan")
    )
    return pd.DataFrame(
        [
            {
                "n_subjects": int(subject_metrics["subject"].nunique()),
                "n_human_records": int(len(record_metrics)),
                "n_channel_comparisons": int(len(channel_metrics)),
                "pairing_mode": config.pairing_mode,
                "amplitude_processing": RAW_AMPLITUDE_MODE,
                "window_size_samples": config.window_size,
                "window_duration_ms": (
                    config.window_size * 1000.0 / config.sample_rate
                ),
                "subject_median_contrast_db": float(np.median(values)),
                "bootstrap_95ci_low_db": confidence_low,
                "bootstrap_95ci_high_db": confidence_high,
                "positive_subjects": positive_count,
                "nonzero_subjects": int(len(nonzero)),
                "exact_sign_test_p": sign_test_p,
                "channel_proportion_exceeding_empty_max": float(
                    channel_metrics["human_exceeds_empty_control_max"].mean()
                ),
                "median_selected_start_ms": float(
                    record_metrics["median_selected_start_ms"].median()
                ),
                "median_interchannel_range_ms": float(
                    record_metrics["selected_start_range_ms"].median()
                ),
            }
        ]
    )


# Existing-result loading and execution
def _load_existing_metrics(
    metrics_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_config = metrics_dir / "config.json"
    if not source_config.exists():
        raise FileNotFoundError(
            f"plot-only requires the source config file: {source_config}"
        )
    payload = json.loads(source_config.read_text(encoding="utf-8"))
    if payload.get("amplitude_processing") != RAW_AMPLITUDE_MODE:
        raise ValueError(
            "The selected metrics were not generated with raw, non-normalized "
            "amplitude processing. Re-run the WAV analysis."
        )
    return (
        _read_csv(metrics_dir / "channel_metrics.csv"),
        _read_csv(metrics_dir / "record_metrics.csv"),
        _read_csv(metrics_dir / "subject_metrics.csv"),
        _read_csv(metrics_dir / "validation_summary.csv"),
    )


def run_human_empty_validation(config: HumanEmptyConfig) -> pd.DataFrame:
    """Execute analysis or regenerate figures from raw non-normalized results."""
    config.validate()
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.results_dir / "requested_config.json", _config_payload(config))

    try:
        if config.plot_only:
            metrics_dir = (config.metrics_dir or config.results_dir).resolve()
            (
                channel_metrics,
                record_metrics,
                subject_metrics,
                summary,
            ) = _load_existing_metrics(metrics_dir)
        else:
            manifest, pairing_audit = build_matched_manifest(config)
            _write_csv(pairing_audit, config.results_dir / "pairing_audit.csv")
            if manifest.empty:
                raise RuntimeError(
                    "No valid matched pair was found. Inspect pairing_audit.csv."
                )
            _write_csv(manifest, config.results_dir / "matched_manifest.csv")

            channel_metrics, errors = process_matched_manifest(manifest, config)
            _write_csv(errors, config.results_dir / "analysis_errors.csv")
            if channel_metrics.empty:
                raise RuntimeError(
                    "No channel comparison completed. Inspect analysis_errors.csv."
                )
            _write_csv(channel_metrics, config.results_dir / "channel_metrics.csv")
            record_metrics, subject_metrics = aggregate_human_empty_metrics(
                channel_metrics
            )
            _write_csv(record_metrics, config.results_dir / "record_metrics.csv")
            _write_csv(subject_metrics, config.results_dir / "subject_metrics.csv")
            summary = make_validation_summary(
                channel_metrics,
                record_metrics,
                subject_metrics,
                config,
            )
            _write_csv(summary, config.results_dir / "validation_summary.csv")

        write_json(config.results_dir / "config.json", _config_payload(config))

        from .human_empty_visualization import create_human_empty_figures

        create_human_empty_figures(
            channel_metrics=channel_metrics,
            record_metrics=record_metrics,
            subject_metrics=subject_metrics,
            summary=summary,
            config=config,
        )
        atomic_write_text(
            config.results_dir / "RUN_COMPLETE.txt",
            "Human-empty validation completed with raw, non-normalized amplitudes.\n",
        )
        return summary
    except Exception as exc:
        atomic_write_text(
            config.results_dir / "ERROR_LOG.txt",
            f"ERROR: {exc!r}\n",
        )
        raise

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Matched human-present versus active empty-room validation using raw "
            "amplitudes and the shared NLOS secondary-window methodology."
        ),
    )
    parser.add_argument("--audio-root", default=str(ROOT_AUDIO_FOLDER))
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS_ROOT / "Human_Empty_Validation"),
    )
    parser.add_argument(
        "--figures-dir",
        default=str(FIGURES_ROOT / "Human_Empty_Validation"),
    )
    parser.add_argument("--surface", default="D")
    parser.add_argument("--scenes", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--postures", default="A,O,C")
    parser.add_argument("--directions", default="ALL")
    parser.add_argument("--transmitters", default="T1,T2,T3,T4,T5,T6,T7,T8,C")
    parser.add_argument(
        "--pairing-mode",
        choices=["same-session", "shared-session"],
        default="same-session",
    )
    parser.add_argument("--empty-reference-session", default=None)
    parser.add_argument("--required-repeats", default="0,1,2")
    parser.add_argument("--max-channels", type=int, default=N_MICROPHONES)
    parser.add_argument("--max-human-files", type=int, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--n-case-figures", type=int, default=10)
    parser.add_argument(
        "--case-selection-mode",
        choices=[
            "subject_balanced",
            "evenly_spaced",
            "median",
            "top",
            "bottom",
            "random",
            "explicit",
        ],
        default="subject_balanced",
    )
    parser.add_argument("--case-pair-ids", default=None)
    parser.add_argument("--case-subjects", default="ALL")
    parser.add_argument("--case-scenes", default="ALL")
    parser.add_argument("--case-transmitters", default="ALL")
    parser.add_argument("--case-channels", default="ALL")
    parser.add_argument("--representative-channel", type=int, default=None)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--metrics-dir", default=None)
    return parser


def config_from_arguments(arguments: argparse.Namespace) -> HumanEmptyConfig:
    directions = _split_tokens(arguments.directions)
    scenes = _split_tokens(arguments.scenes, uppercase=False)
    postures = _split_tokens(arguments.postures)
    transmitters = _split_tokens(arguments.transmitters)
    required_repeats = _split_tokens(arguments.required_repeats, uppercase=False)
    return HumanEmptyConfig(
        audio_root=Path(arguments.audio_root).expanduser().resolve(),
        results_dir=Path(arguments.results_dir).expanduser().resolve(),
        figures_dir=Path(arguments.figures_dir).expanduser().resolve(),
        surface=str(arguments.surface).upper(),
        scenes=scenes or tuple(str(index) for index in range(10)),
        postures=postures or ("A", "O", "C"),
        directions=directions,
        transmitters=transmitters
        or ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "C"),
        pairing_mode=arguments.pairing_mode,
        empty_reference_session=arguments.empty_reference_session,
        required_repeats=required_repeats or DEFAULT_REQUIRED_REPEATS,
        max_channels=arguments.max_channels,
        max_human_files=arguments.max_human_files,
        bootstrap_iterations=arguments.bootstrap_iterations,
        random_seed=arguments.random_seed,
        n_case_figures=arguments.n_case_figures,
        case_selection_mode=arguments.case_selection_mode,
        case_pair_ids=_split_tokens(arguments.case_pair_ids, uppercase=False),
        case_subjects=_split_tokens(arguments.case_subjects),
        case_scenes=_split_tokens(arguments.case_scenes, uppercase=False),
        case_transmitters=_split_tokens(arguments.case_transmitters),
        case_channels=_split_channels(arguments.case_channels),
        representative_channel=arguments.representative_channel,
        plot_only=bool(arguments.plot_only),
        metrics_dir=(
            Path(arguments.metrics_dir).expanduser().resolve()
            if arguments.metrics_dir
            else None
        ),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    config = config_from_arguments(arguments)
    try:
        summary = run_human_empty_validation(config)
        print("Human-empty validation completed successfully.")
        print(f"Results: {config.results_dir}")
        print(f"Figures: {config.figures_dir}")
        print(summary.to_string(index=False))
        return 0
    except Exception as exc:
        print(f"HUMAN-EMPTY VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "HumanEmptyConfig",
    "HumanEmptyRecord",
    "RAW_AMPLITUDE_MODE",
    "aggregate_human_empty_metrics",
    "build_matched_manifest",
    "main",
    "make_validation_summary",
    "parse_human_empty_record",
    "process_matched_manifest",
    "read_wav_raw",
    "run_human_empty_validation",
]
