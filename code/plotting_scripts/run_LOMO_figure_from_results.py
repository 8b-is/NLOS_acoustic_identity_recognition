# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOMO_RESULTS_CSV: Path | None = None

# Example:
# LOMO_RESULTS_CSV = Path(
#     r"D:\...\TSET_T1_T2_T3_T4_T5_T6_T7_T8_M\LOMO_results.csv"
# )

OUTPUT_FILE_NAME = "cross_material_lomo_accuracy.png"
OUTPUT_DPI = 300
SHOW_FIGURE = True

MATERIAL_ORDER = ["Wood", "Iron", "Brick", "White appliance", "PVC"]
FEATURE_ORDER = ["LPVG_D0", "STFT", "Fusion_D0"]
FEATURE_LABELS = {
    "LPVG_D0": "LPVG (d=0)",
    "STFT": "STFT",
    "Fusion_D0": "Fusion (d=0)",
}


# =============================================================================
# FILE AND COLUMN HELPERS
# =============================================================================
def _candidate_result_paths() -> Iterable[Path]:
    output_root = Path(
        os.environ.get(
            "NLOS_OUTPUT_ROOT",
            "NLOS_identity_recognition_results",
        )
    )

    yield Path("LOMO_results.csv")
    yield (
        output_root
        / "cross_material"
        / "TSET_T1_T2_T3_T4_T5_T6_T7_T8_M"
        / "LOMO_results.csv"
    )

    for root_name in (
        "NLOS_identity_recognition_publication_results",
        "NLOS_identity_recognition_results",
    ):
        yield (
            Path(root_name)
            / "cross_material"
            / "TSET_T1_T2_T3_T4_T5_T6_T7_T8_M"
            / "LOMO_results.csv"
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
        "LOMO_results.csv was not found. Set LOMO_RESULTS_CSV at the top "
        "of this script."
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
        "white goods": "White appliance",
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


def create_lomo_bar_chart(results_csv: Path) -> Path:
    dataframe = _read_csv_auto(results_csv)

    feature_column = _find_column(
        dataframe,
        ["feature", "feature_name", "feature_representation"],
    )
    material_column = _find_column(
        dataframe,
        ["target_material", "held_out_material", "test_material", "material"],
    )
    accuracy_column = _find_column(
        dataframe,
        ["accuracy_percent", "accuracy", "acc", "test_accuracy"],
    )

    plot_data = dataframe.copy()
    plot_data["_feature"] = plot_data[feature_column].map(_canonical_feature)
    plot_data["_material"] = plot_data[material_column].map(_canonical_material)
    plot_data["_accuracy"] = _accuracy_to_percent(plot_data[accuracy_column])

    pivot = plot_data.pivot_table(
        index="_material",
        columns="_feature",
        values="_accuracy",
        aggfunc="mean",
    ).reindex(index=MATERIAL_ORDER, columns=FEATURE_ORDER)

    if pivot.isna().any().any():
        missing = [
            f"{material}/{feature}"
            for material in MATERIAL_ORDER
            for feature in FEATURE_ORDER
            if not np.isfinite(pivot.loc[material, feature])
        ]
        raise ValueError(
            "LOMO result combinations are incomplete: " + ", ".join(missing)
        )

    values = pivot.to_numpy(dtype=float)
    x_positions = np.arange(len(MATERIAL_ORDER), dtype=float)
    bar_width = 0.24

    figure, axis = plt.subplots(figsize=(10.8, 5.8), dpi=OUTPUT_DPI)

    for feature_index, feature_name in enumerate(FEATURE_ORDER):
        positions = x_positions + (feature_index - 1) * bar_width
        bars = axis.bar(
            positions,
            values[:, feature_index],
            width=bar_width,
            label=FEATURE_LABELS[feature_name],
        )

        for bar, value in zip(bars, values[:, feature_index]):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1.0,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10.5,
            )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(MATERIAL_ORDER, fontsize=10)
    axis.set_xlabel("Held-out test material", fontsize=12, labelpad=14)
    axis.set_ylabel("Accuracy (%)", fontsize=12)
    axis.set_ylim(0.0, max(105.0, float(np.nanmax(values)) + 10.0))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper center", ncol=3, frameon=True, fontsize=10)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()

    output_path = results_csv.parent / OUTPUT_FILE_NAME
    figure.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(figure)

    return output_path


def main() -> None:
    results_csv = _resolve_results_csv(LOMO_RESULTS_CSV)
    output_path = create_lomo_bar_chart(results_csv)
    print(f"Input CSV : {results_csv}")
    print(f"Figure    : {output_path}")


if __name__ == "__main__":
    main()
