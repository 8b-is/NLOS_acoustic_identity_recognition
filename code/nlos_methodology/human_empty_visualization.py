# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .metadata import PUBLIC_SUBJECT_ORDER

if TYPE_CHECKING:
    from .human_empty_validation import HumanEmptyConfig


# Publication style
TITLE_FONTSIZE = 19
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 13

CASE_FIGSIZE = (12.0, 9.4)
SUBJECT_FIGSIZE = (12.0, 6.6)
FIGURE_DPI = 600

LINE_WIDTH = 1.6
MEAN_LINE_WIDTH = 1.8
MARKER_SIZE = 10
MEDIAN_LINE_WIDTH = 2.0
ZERO_LINE_WIDTH = 1.4
SELECTED_WINDOW_ALPHA = 0.12
EMPTY_RANGE_ALPHA = 0.18
CI_ALPHA = 0.16

ABSENT_COLOR = "#1f77b4"
PRESENT_COLOR = "#d62728"
WINDOW_COLOR = "#ff7f0e"
REPEAT_COLOR = "#bdbdbd"
RANGE_COLOR = "#bdbdbd"


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": TICK_LABEL_FONTSIZE,
            "axes.titlesize": TITLE_FONTSIZE,
            "axes.labelsize": AXIS_LABEL_FONTSIZE,
            "xtick.labelsize": TICK_LABEL_FONTSIZE,
            "ytick.labelsize": TICK_LABEL_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _subject_sort_key(label: str) -> Tuple[int, str]:
    text = str(label)
    try:
        return PUBLIC_SUBJECT_ORDER.index(text), text
    except ValueError:
        suffix = text[1:] if text.upper().startswith("S") else ""
        return (
            int(suffix) + len(PUBLIC_SUBJECT_ORDER)
            if suffix.isdigit()
            else 10**9,
            text,
        )


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.8)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


# ---------------------------------------------------------------------------
# Case selection
# ---------------------------------------------------------------------------
def _filter_case_candidates(
    channel_metrics: pd.DataFrame,
    config: "HumanEmptyConfig",
) -> pd.DataFrame:
    dataframe = channel_metrics.copy()
    if config.case_subjects:
        dataframe = dataframe[
            dataframe["subject"].astype(str).isin(config.case_subjects)
        ]
    if config.case_scenes:
        dataframe = dataframe[
            dataframe["scene"].astype(str).isin(config.case_scenes)
        ]
    if config.case_transmitters:
        dataframe = dataframe[
            dataframe["transmitter"].astype(str).isin(
                config.case_transmitters
            )
        ]
    if config.case_channels:
        dataframe = dataframe[
            dataframe["channel"].astype(int).isin(config.case_channels)
        ]
    return dataframe


def _select_channel(
    channels: pd.DataFrame,
    config: "HumanEmptyConfig",
    target_contrast: Optional[float] = None,
) -> pd.Series:
    candidates = channels.copy()
    if config.representative_channel is not None:
        requested = candidates[
            candidates["channel"].astype(int)
            == int(config.representative_channel)
        ]
        if not requested.empty:
            return requested.sort_values("channel").iloc[0]

    if target_contrast is None:
        target_contrast = float(
            candidates["contrast_vs_empty_median_db"].median()
        )

    candidates["distance"] = (
        candidates["contrast_vs_empty_median_db"] - target_contrast
    ).abs()
    return candidates.sort_values(["distance", "channel"]).iloc[0]


def _select_case_rows(
    channel_metrics: pd.DataFrame,
    record_metrics: pd.DataFrame,
    config: "HumanEmptyConfig",
) -> List[pd.Series]:
    candidates = _filter_case_candidates(channel_metrics, config)
    if candidates.empty:
        candidates = channel_metrics.copy()

    pair_ids = candidates["pair_id"].drop_duplicates().tolist()
    records = record_metrics[
        record_metrics["pair_id"].isin(pair_ids)
    ].copy()
    if records.empty:
        records = record_metrics.copy()

    target_count = max(1, int(config.n_case_figures))
    selected_pair_ids: List[str] = []
    mode = config.case_selection_mode

    if mode == "explicit" and config.case_pair_ids:
        valid = set(records["pair_id"].astype(str))
        selected_pair_ids = [
            item for item in config.case_pair_ids if item in valid
        ]
    elif mode == "subject_balanced":
        subjects = sorted(
            records["subject"].astype(str).unique(),
            key=_subject_sort_key,
        )
        for subject in subjects:
            group = records[
                records["subject"].astype(str) == subject
            ].copy()
            target = float(group["median_contrast_db"].median())
            chosen = group.assign(
                distance=(group["median_contrast_db"] - target).abs()
            ).sort_values(["distance", "pair_id"]).iloc[0]
            selected_pair_ids.append(str(chosen["pair_id"]))

        if len(selected_pair_ids) > target_count:
            indices = np.unique(
                np.round(
                    np.linspace(
                        0,
                        len(selected_pair_ids) - 1,
                        target_count,
                    )
                ).astype(int)
            )
            selected_pair_ids = [
                selected_pair_ids[index] for index in indices
            ]
    elif mode == "top":
        selected_pair_ids = (
            records.sort_values(
                ["median_contrast_db", "pair_id"],
                ascending=[False, True],
            )["pair_id"]
            .astype(str)
            .head(target_count)
            .tolist()
        )
    elif mode == "bottom":
        selected_pair_ids = (
            records.sort_values(
                ["median_contrast_db", "pair_id"],
                ascending=[True, True],
            )["pair_id"]
            .astype(str)
            .head(target_count)
            .tolist()
        )
    elif mode == "random":
        selected_pair_ids = (
            records.sample(
                n=min(target_count, len(records)),
                random_state=config.random_seed,
            )["pair_id"]
            .astype(str)
            .tolist()
        )
    elif mode == "median":
        target = float(records["median_contrast_db"].median())
        selected_pair_ids = (
            records.assign(
                distance=(records["median_contrast_db"] - target).abs()
            )
            .sort_values(["distance", "pair_id"])["pair_id"]
            .astype(str)
            .head(target_count)
            .tolist()
        )
    else:
        ordered = records.sort_values(
            ["median_contrast_db", "pair_id"]
        ).reset_index(drop=True)
        if len(ordered) <= target_count:
            selected_pair_ids = ordered["pair_id"].astype(str).tolist()
        else:
            indices = np.unique(
                np.round(
                    np.linspace(
                        0,
                        len(ordered) - 1,
                        target_count,
                    )
                ).astype(int)
            )
            selected_pair_ids = (
                ordered.iloc[indices]["pair_id"].astype(str).tolist()
            )

    if len(selected_pair_ids) < target_count:
        remaining = [
            str(item)
            for item in records["pair_id"].tolist()
            if str(item) not in selected_pair_ids
        ]
        selected_pair_ids.extend(
            remaining[: target_count - len(selected_pair_ids)]
        )

    record_index = record_metrics.set_index("pair_id", drop=False)
    rows: List[pd.Series] = []
    for pair_id in selected_pair_ids[:target_count]:
        channels = candidates[
            candidates["pair_id"].astype(str) == pair_id
        ]
        if channels.empty:
            channels = channel_metrics[
                channel_metrics["pair_id"].astype(str) == pair_id
            ]
        if channels.empty:
            continue
        target = (
            float(record_index.loc[pair_id, "median_contrast_db"])
            if pair_id in record_index.index
            else None
        )
        rows.append(_select_channel(channels, config, target))
    return rows


# Raw case data
def _load_case_data(
    row: pd.Series,
    config: "HumanEmptyConfig",
) -> Dict[str, object]:
    from .human_empty_validation import read_wav_raw
    from .preprocessing import extract_window_by_index

    human = read_wav_raw(
        Path(str(row["human_wav"])),
        config.sample_rate,
    )
    empty_paths = [
        Path(item)
        for item in str(row["empty_wavs"]).split("|")
        if item
    ]
    empties = [
        read_wav_raw(path, config.sample_rate)
        for path in empty_paths
    ]

    channel_index = int(row["channel"]) - 1
    selected_index = int(row["selected_window_index"])
    start_sample = selected_index * config.window_size

    human_channel = np.asarray(
        human[:, channel_index],
        dtype=np.float64,
    )
    empty_channels = [
        np.asarray(empty[:, channel_index], dtype=np.float64)
        for empty in empties
    ]

    common_length = min(
        len(human_channel),
        *(len(item) for item in empty_channels),
    )
    human_full = human_channel[:common_length]
    empty_full = [
        item[:common_length] for item in empty_channels
    ]
    empty_mean_full = np.mean(
        np.stack(empty_full, axis=0),
        axis=0,
    )

    human_raw_segment = extract_window_by_index(
        human_channel,
        selected_index,
        config.window_size,
    )
    if human_raw_segment is None:
        raise ValueError(
            f"Could not extract selected human window for pair {row['pair_id']}."
        )
    human_segment = np.asarray(
        human_raw_segment,
        dtype=np.float64,
    )

    empty_segments: List[np.ndarray] = []
    for empty_channel in empty_channels:
        empty_raw_segment = extract_window_by_index(
            empty_channel,
            selected_index,
            config.window_size,
        )
        if empty_raw_segment is None:
            raise ValueError(
                f"Could not extract selected empty window for pair {row['pair_id']}."
            )
        empty_segments.append(
            np.asarray(empty_raw_segment, dtype=np.float64)
        )

    empty_stack = np.stack(empty_segments, axis=0)

    return {
        "human_full": human_full,
        "empty_full": empty_full,
        "empty_mean_full": empty_mean_full,
        "human_segment": human_segment,
        "empty_segment_mean": np.mean(empty_stack, axis=0),
        "empty_segment_min": np.min(empty_stack, axis=0),
        "empty_segment_max": np.max(empty_stack, axis=0),
        "start_sample": start_sample,
    }


def _case_figure_stem(
    order_index: int,
    row: pd.Series,
) -> str:
    return (
        f"figure_case_{order_index:02d}_"
        f"{row['subject']}_sc{row['scene']}_"
        f"{row['transmitter']}_ch{int(row['channel'])}"
    )


def _finite_limit(arrays: List[np.ndarray]) -> float:
    values = [
        np.asarray(array, dtype=np.float64).reshape(-1)
        for array in arrays
        if np.asarray(array).size
    ]
    if not values:
        return 1.0
    maximum = max(float(np.max(np.abs(value))) for value in values)
    if not np.isfinite(maximum) or maximum <= 0:
        return 1.0
    return maximum * 1.08


def _create_case_figure(
    row: pd.Series,
    config: "HumanEmptyConfig",
    order_index: int,
) -> Tuple[str, str]:
    case = _load_case_data(row, config)

    full_time_ms = (
        np.arange(len(case["human_full"]), dtype=np.float64)
        * 1000.0
        / config.sample_rate
    )
    start_ms = (
        int(case["start_sample"])
        * 1000.0
        / config.sample_rate
    )
    end_ms = start_ms + (
        config.window_size
        * 1000.0
        / config.sample_rate
    )
    selected_time_ms = start_ms + (
        np.arange(config.window_size, dtype=np.float64)
        * 1000.0
        / config.sample_rate
    )

    common_limit = _finite_limit(
        [
            case["human_full"],
            case["empty_mean_full"],
            *case["empty_full"],
            case["human_segment"],
            case["empty_segment_mean"],
            case["empty_segment_min"],
            case["empty_segment_max"],
        ]
    )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=CASE_FIGSIZE,
    )

    for index, repeat in enumerate(case["empty_full"]):
        axes[0].plot(
            full_time_ms,
            repeat,
            color=REPEAT_COLOR,
            linewidth=0.9,
            alpha=0.55,
            label="Subject-absent repeats" if index == 0 else None,
        )
    axes[0].plot(
        full_time_ms,
        case["empty_mean_full"],
        color=ABSENT_COLOR,
        linewidth=MEAN_LINE_WIDTH,
        label="Subject-absent mean",
    )
    axes[0].axvspan(
        start_ms,
        end_ms,
        color=WINDOW_COLOR,
        alpha=SELECTED_WINDOW_ALPHA,
        label="Selected window",
    )
    axes[0].set_title(
        "Matched subject-absent recordings",
        pad=12,
    )
    axes[0].set_ylabel("Amplitude")
    axes[0].set_ylim(-common_limit, common_limit)
    axes[0].legend(
        frameon=False,
        ncol=3,
        loc="upper center",
    )
    _style_axis(axes[0])

    axes[1].plot(
        full_time_ms,
        case["human_full"],
        color=PRESENT_COLOR,
        linewidth=LINE_WIDTH,
        label="Subject present",
    )
    axes[1].axvspan(
        start_ms,
        end_ms,
        color=WINDOW_COLOR,
        alpha=SELECTED_WINDOW_ALPHA,
        label="Selected window",
    )
    axes[1].set_title(
        "Matched subject-present recording",
        pad=12,
    )
    axes[1].set_ylabel("Amplitude")
    axes[1].set_ylim(-common_limit, common_limit)
    axes[1].legend(
        frameon=False,
        ncol=2,
        loc="upper right",
    )
    _style_axis(axes[1])

    axes[2].fill_between(
        selected_time_ms,
        case["empty_segment_min"],
        case["empty_segment_max"],
        color=RANGE_COLOR,
        alpha=EMPTY_RANGE_ALPHA,
        label="Subject-absent range",
    )
    axes[2].plot(
        selected_time_ms,
        case["empty_segment_mean"],
        color=ABSENT_COLOR,
        linewidth=MEAN_LINE_WIDTH,
        label="Subject-absent mean",
    )
    axes[2].plot(
        selected_time_ms,
        case["human_segment"],
        color=PRESENT_COLOR,
        linewidth=LINE_WIDTH,
        label="Subject present",
    )
    axes[2].set_title(
        "Selected secondary-echo window "
        f"({start_ms:.2f}–{end_ms:.2f} ms)",
        pad=12,
    )
    axes[2].set_xlabel("Recording time (ms)")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_ylim(-common_limit, common_limit)
    axes[2].legend(
        frameon=False,
        ncol=3,
        loc="upper center",
    )
    _style_axis(axes[2])

    figure.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.075,
        top=0.97,
        hspace=0.46,
    )

    stem = _case_figure_stem(order_index, row)
    output_path = config.figures_dir / f"{stem}.png"
    _save_figure(figure, output_path)

    caption = (
        "Matched subject-absent and subject-present raw-amplitude recordings. "
        "The shaded interval is selected from the subject-present recording by "
        "the original non-overlapping energy-window rule and the identical "
        "recording-time interval is extracted from all matched subject-absent "
        "repeats. No amplitude normalization is applied."
    )
    return stem, caption


# Subject-level contrast figure
def _create_subject_summary_figure(
    subject_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    config: "HumanEmptyConfig",
) -> Tuple[str, str]:
    summary_row = summary.iloc[0]
    ordered = subject_metrics.sort_values(
        "subject",
        key=lambda series: series.map(_subject_sort_key),
    ).reset_index(drop=True)

    x = np.arange(len(ordered))
    y = ordered["median_contrast_db"].to_numpy(dtype=float)

    median_value = float(
        summary_row["subject_median_contrast_db"]
    )
    ci_low = float(
        summary_row["bootstrap_95ci_low_db"]
    )
    ci_high = float(
        summary_row["bootstrap_95ci_high_db"]
    )

    if not np.isfinite(ci_low):
        ci_low = median_value
    if not np.isfinite(ci_high):
        ci_high = median_value

    finite_values = np.concatenate(
        [
            y[np.isfinite(y)],
            np.asarray(
                [median_value, ci_low, ci_high, 0.0],
                dtype=np.float64,
            ),
        ]
    )
    y_max = float(np.max(finite_values))
    y_min = float(np.min(finite_values))
    y_range = max(1.0, y_max - y_min)
    y_top = y_max + 0.08 * y_range
    y_bottom = min(-0.5, y_min - 0.04 * y_range)

    figure, axis = plt.subplots(
        figsize=SUBJECT_FIGSIZE,
    )

    axis.axhspan(
        ci_low,
        ci_high,
        color=ABSENT_COLOR,
        alpha=CI_ALPHA,
        zorder=0,
    )
    axis.axhline(
        median_value,
        color=ABSENT_COLOR,
        linewidth=MEDIAN_LINE_WIDTH,
        zorder=1,
    )
    axis.axhline(
        0.0,
        color=ABSENT_COLOR,
        linestyle="--",
        linewidth=ZERO_LINE_WIDTH,
        zorder=1,
    )
    axis.plot(
        x,
        y,
        color=ABSENT_COLOR,
        marker="o",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        zorder=3,
    )

    axis.set_xlabel("Subjects")
    axis.set_ylabel(
        "Human / empty room - control contrast (dB)"
    )
    axis.set_xticks(x)
    axis.set_xticklabels(
        ordered["subject"].astype(str).tolist()
    )
    axis.set_ylim(y_bottom, y_top)
    _style_axis(axis)

    handles = [
        Line2D(
            [0],
            [0],
            color=ABSENT_COLOR,
            linewidth=MEDIAN_LINE_WIDTH,
            label=f"Median ({median_value:.2f} dB)",
        ),
        Patch(
            facecolor=ABSENT_COLOR,
            alpha=CI_ALPHA,
            label=f"95% CI [{ci_low:.2f}, {ci_high:.2f}] dB",
        ),
        Line2D(
            [0],
            [0],
            color=ABSENT_COLOR,
            linestyle="--",
            linewidth=ZERO_LINE_WIDTH,
            label="Zero contrast (0 dB)",
        ),
    ]
    axis.legend(
        handles=handles,
        frameon=False,
        loc="upper right",
    )

    figure.tight_layout()

    stem = "figure_subject_contrast_publication"
    output_path = config.figures_dir / f"{stem}.png"
    _save_figure(figure, output_path)

    legacy_path = (
        config.figures_dir
        / "figure_02_subject_level_summary.png"
    )
    shutil.copyfile(output_path, legacy_path)

    caption = (
        "Subject-level median matched selected-window contrast. The horizontal "
        "solid line is the across-subject median, the shaded band is its "
        "bootstrap 95% confidence interval, and the dashed line denotes zero "
        "contrast. No amplitude normalization is used."
    )
    return stem, caption


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def create_human_empty_figures(
    channel_metrics: pd.DataFrame,
    record_metrics: pd.DataFrame,
    subject_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    config: "HumanEmptyConfig",
) -> None:
    """Generate publication figures from computed result tables."""
    _set_style()
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, object]] = []
    selected_case_rows: List[Dict[str, object]] = []

    selected_rows = _select_case_rows(
        channel_metrics,
        record_metrics,
        config,
    )
    for order_index, row in enumerate(
        selected_rows,
        start=1,
    ):
        stem, caption = _create_case_figure(
            row,
            config,
            order_index,
        )
        manifest_rows.append(
            {
                "figure_stem": stem,
                "file": f"{stem}.png",
                "caption": caption,
            }
        )
        selected_case_rows.append(
            {
                "figure_order": order_index,
                "figure_stem": stem,
                "pair_id": row["pair_id"],
                "subject": row["subject"],
                "scene": row["scene"],
                "scene_label": row["scene_label"],
                "posture": row["posture"],
                "direction": row["direction"],
                "transmitter": row["transmitter"],
                "channel": int(row["channel"]),
                "selected_start_ms": float(
                    row["selected_start_ms"]
                ),
                "selected_end_ms": float(
                    row["selected_end_ms"]
                ),
                "contrast_vs_empty_median_db": float(
                    row["contrast_vs_empty_median_db"]
                ),
                "contrast_vs_empty_max_db": float(
                    row["contrast_vs_empty_max_db"]
                ),
            }
        )

    subject_stem, subject_caption = (
        _create_subject_summary_figure(
            subject_metrics,
            summary,
            config,
        )
    )
    manifest_rows.append(
        {
            "figure_stem": subject_stem,
            "file": f"{subject_stem}.png",
            "caption": subject_caption,
        }
    )

    pd.DataFrame(selected_case_rows).to_csv(
        config.results_dir / "selected_figure_cases.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(manifest_rows).to_csv(
        config.results_dir / "figure_manifest.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )


__all__ = ["create_human_empty_figures"]
