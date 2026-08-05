# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

TRANSFER_RESULTS_CSV: Path | None = None

# Example:
# TRANSFER_RESULTS_CSV = Path(
#     r"D:\...\TSET_T1_T2_T3_T4_T5_T6_T7_T8_M"
#     r"\single_source_transfer_results.csv"
# )

OUTPUT_FILE_NAME = "cross_material_transfer_matrices.png"
OUTPUT_DPI = 300
SHOW_FIGURE = True

MATERIAL_ORDER = ["Wood", "Iron", "Brick", "White appliance", "PVC"]
FEATURE_ORDER = ["LPVG_D0", "STFT", "Fusion_D0"]
FEATURE_TITLES = {
    "LPVG_D0": "LPVG (d=0)",
    "STFT": "STFT",
    "Fusion_D0": "Fusion (d=0)",
}


def _candidate_result_paths() -> Iterable[Path]:
    output_root = Path(
        os.environ.get(
            "NLOS_OUTPUT_ROOT",
            "NLOS_identity_recognition_results",
        )
    )

    yield Path("single_source_transfer_results.csv")
    yield (
        output_root
        / "cross_material"
        / "TSET_T1_T2_T3_T4_T5_T6_T7_T8_M"
        / "single_source_transfer_results.csv"
    )

    for root_name in (
        "NLOS_identity_recognition_publication_results",
        "NLOS_identity_recognition_results",
    ):
        yield (
            Path(root_name)
            / "cross_material"
            / "TSET_T1_T2_T3_T4_T5_T6_T7_T8_M"
            / "single_source_transfer_results.csv"
        )


def _resolve_results_csv(configured_path: Path | None) -> Path:
    if configured_path is not None:
        path = Path(configured_path)
        if not path.exists():
            raise FileNotFoundError(f"Results CSV was not found: {path}")
        return path.resolve()

    for candidate in _candidate_result_paths():
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "single_source_transfer_results.csv was not found. "
    )


def _read_csv_auto(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for separator in (";", ",", "\t"):
        try:
            dataframe = pd.read_csv(
                path,
                sep=separator,
                encoding="utf-8-sig",
            )
            if dataframe.shape[1] > 1:
                return dataframe
        except Exception as exc:  
            last_error = exc

    raise ValueError(f"Could not read result CSV {path}: {last_error}")


def _find_column(dataframe: pd.DataFrame, aliases: Iterable[str]) -> str:
    normalized = {
        str(column).strip().lower().replace(" ", "_"): str(column)
        for column in dataframe.columns
    }
    for alias in aliases:
        key = alias.strip().lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    raise KeyError(
        f"Required column was not found. Expected one of {list(aliases)}; "
        f"available columns are {list(dataframe.columns)}."
    )


def _canonical_feature(value: object) -> str:
    text = str(value).strip().lower().replace(" ", "")
    text = text.replace("(d=0)", "d0").replace("_deg_", "_")

    if "fusion" in text and ("d0" in text or "d=0" in text):
        return "Fusion_D0"
    if "lpvg" in text and ("d0" in text or "d=0" in text):
        return "LPVG_D0"
    if text in {"stft"} or ("stft" in text and "fusion" not in text):
        return "STFT"
    return str(value).strip()


def _canonical_material(value: object) -> str:
    text = str(value).strip()
    compact = text.lower().replace("_", " ").replace("-", " ")
    compact = " ".join(compact.split())

    aliases = {
        "wood": "Wood",
        "iron": "Iron",
        "brick": "Brick",
        "white appliance": "White appliance",
        "white appliance surface": "White appliance",
        "pvc": "PVC",
        "pvc panel": "PVC",
    }
    return aliases.get(compact, text)


def _accuracy_to_percent(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        raise ValueError("The accuracy column contains no numeric values.")
    if float(finite.max()) <= 1.0 + 1e-9:
        numeric = numeric * 100.0
    return numeric


# =============================================================================
# FIGURE
# =============================================================================
def create_transfer_heatmaps(results_csv: Path) -> Path:
    dataframe = _read_csv_auto(results_csv)

    feature_column = _find_column(
        dataframe,
        ["feature", "feature_name", "feature_representation"],
    )
    source_column = _find_column(
        dataframe,
        ["source_material", "training_material", "train_material"],
    )
    target_column = _find_column(
        dataframe,
        ["target_material", "test_material", "held_out_material"],
    )
    accuracy_column = _find_column(
        dataframe,
        ["accuracy_percent", "accuracy", "acc", "test_accuracy"],
    )

    plot_data = dataframe.copy()
    plot_data["_feature"] = plot_data[feature_column].map(_canonical_feature)
    plot_data["_source"] = plot_data[source_column].map(_canonical_material)
    plot_data["_target"] = plot_data[target_column].map(_canonical_material)
    plot_data["_accuracy"] = _accuracy_to_percent(plot_data[accuracy_column])

    palette = ["#16324F", "#2A628F", "#4EA5D9", "#7FC8A9", "#F6D55C"]
    colormap = LinearSegmentedColormap.from_list(
        "cross_material_accuracy",
        palette,
        N=256,
    )
    colormap.set_bad(color="#ECECEC")
    normalization = Normalize(vmin=0.0, vmax=100.0)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12.6, 4.8),
        dpi=OUTPUT_DPI,
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.88,
        bottom=0.25,
        top=0.88,
        wspace=0.14,
    )

    image = None
    for panel_index, (axis, feature_name) in enumerate(
        zip(axes, FEATURE_ORDER)
    ):
        selected = plot_data.loc[plot_data["_feature"] == feature_name]
        if selected.empty:
            raise ValueError(
                f"No rows were found for {feature_name}. Values seen in the "
                f"feature column: {sorted(plot_data['_feature'].unique())}"
            )

        matrix_frame = selected.pivot_table(
            index="_source",
            columns="_target",
            values="_accuracy",
            aggfunc="mean",
        ).reindex(index=MATERIAL_ORDER, columns=MATERIAL_ORDER)

        matrix = matrix_frame.to_numpy(dtype=float)
        np.fill_diagonal(matrix, np.nan)

        image = axis.imshow(
            matrix,
            cmap=colormap,
            norm=normalization,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(FEATURE_TITLES[feature_name], fontsize=13, pad=10)
        axis.set_xticks(np.arange(len(MATERIAL_ORDER)))
        axis.set_yticks(np.arange(len(MATERIAL_ORDER)))
        axis.set_xticklabels(
            MATERIAL_ORDER,
            rotation=35,
            ha="right",
            fontsize=9,
        )
        axis.set_yticklabels(
            MATERIAL_ORDER if panel_index == 0 else [""] * len(MATERIAL_ORDER),
            fontsize=9,
        )
        axis.set_xlabel("Test material", fontsize=11)
        if panel_index == 0:
            axis.set_ylabel("Training material", fontsize=11)

        axis.set_xticks(np.arange(-0.5, len(MATERIAL_ORDER), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(MATERIAL_ORDER), 1), minor=True)
        axis.grid(which="minor", linewidth=1.0, color="white")
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.tick_params(axis="both", which="major", length=3)

        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if not np.isfinite(value):
                    continue
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="black",
                )

        for spine in axis.spines.values():
            spine.set_visible(False)

    if image is None:  
        raise RuntimeError("No heatmap panel was generated.")

    colorbar_axis = figure.add_axes([0.90, 0.18, 0.022, 0.68])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Identity accuracy (%)", fontsize=11, labelpad=10)
    colorbar.ax.tick_params(labelsize=9)

    output_path = results_csv.parent / OUTPUT_FILE_NAME
    figure.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(figure)

    return output_path


def main() -> None:
    results_csv = _resolve_results_csv(TRANSFER_RESULTS_CSV)
    output_path = create_transfer_heatmaps(results_csv)
    print(f"Input CSV : {results_csv}")
    print(f"Figure    : {output_path}")


if __name__ == "__main__":
    main()
