# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES_ROOT
from .metadata import SCENE_DESCRIPTIONS


# Fixed manuscript figure style
FONT_SIZE_DELTA = 5

FIGURE_SIZE = (7.2, 6.5)
DISPLAY_DPI = 300
SAVE_DPI = 300
COLOR_MAP = "YlGnBu"

TITLE_FONT_SIZE = 14 + FONT_SIZE_DELTA
AXIS_LABEL_FONT_SIZE = 14 + FONT_SIZE_DELTA
TICK_FONT_SIZE = 11 + FONT_SIZE_DELTA
DIAGONAL_CELL_FONT_SIZE = 11 + FONT_SIZE_DELTA
OFF_DIAGONAL_CELL_FONT_SIZE = 13 + FONT_SIZE_DELTA
COLORBAR_TICK_FONT_SIZE = 12

X_TICK_ROTATION = 45
GRID_LINE_STYLE = "-"
GRID_LINE_WIDTH = 1.0
GRID_ALPHA = 0.4

SUBPLOT_RIGHT = 0.88
COLORBAR_AXES = [0.90, 0.15, 0.022, 0.70]


# Data loading and validation
def load_confusion_matrix(
    csv_path: Path | str,
) -> Tuple[np.ndarray, list[str], list[str]]:
    """Load a labeled square confusion matrix without modifying it."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            "Confusion-matrix CSV was not found:\n"
            f"{path}"
        )

    dataframe = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    if dataframe.empty:
        raise ValueError(
            f"The selected confusion-matrix CSV is empty:\n{path}"
        )

    try:
        dataframe = dataframe.apply(pd.to_numeric, errors="raise")
    except Exception as exc:
        raise ValueError(
            "All confusion-matrix cells must contain numeric values."
        ) from exc

    matrix = dataframe.to_numpy(dtype=np.float64)
    row_labels = [str(label) for label in dataframe.index.tolist()]
    column_labels = [str(label) for label in dataframe.columns.tolist()]

    if matrix.ndim != 2:
        raise ValueError(
            f"The confusion matrix must be two-dimensional: {matrix.shape}"
        )
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"The confusion matrix must be square: {matrix.shape}"
        )
    if len(row_labels) != matrix.shape[0]:
        raise ValueError(
            "The number of row labels does not match the matrix."
        )
    if len(column_labels) != matrix.shape[1]:
        raise ValueError(
            "The number of column labels does not match the matrix."
        )
    if row_labels != column_labels:
        raise ValueError(
            "Row and column labels are not identical and in the same order."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "The confusion matrix contains NaN or infinite values."
        )
    if np.any(matrix < 0):
        raise ValueError(
            "The confusion matrix contains negative values."
        )

    return matrix, row_labels, column_labels


# Titles and filenames
def create_automatic_title(csv_path: Path | str) -> str:
    """Generate the publication title from a CM filename."""
    stem = Path(csv_path).stem

    if stem == "LOSO_overall_CM":
        return "Overall Confusion Matrix"

    scene_prefix = "LOSO_scene_"
    if stem.startswith(scene_prefix):
        scene_code = stem[len(scene_prefix):].split("_")[0]
        description = SCENE_DESCRIPTIONS.get(
            scene_code,
            f"Scene {scene_code}",
        )
        if description == "No object":
            return "No Objects Configuration"
        if description == "PVC":
            return "PVC Panel Configuration"
        return f"{description.title()} Configuration"

    return "Confusion Matrix"


def sanitize_filename_component(text: str) -> str:
    """Return a filesystem-safe filename component."""
    safe_text = "".join(
        character
        if character.isalnum() or character in {"-", "_"}
        else "_"
        for character in str(text)
    )
    while "__" in safe_text:
        safe_text = safe_text.replace("__", "_")
    return safe_text.strip("_")


def infer_experiment_components(
    csv_path: Path | str,
) -> Tuple[str, str, str]:

    path = Path(csv_path)
    classifier_name = path.parent.name or "Unknown_Classifier"
    feature_name = path.parent.parent.name or "Unknown_Feature"
    tset_name = path.parent.parent.parent.name or "Unknown_TSET"

    if not tset_name.startswith("TSET_"):
        tset_name = "Unknown_TSET"

    return (
        sanitize_filename_component(tset_name),
        sanitize_filename_component(feature_name),
        sanitize_filename_component(classifier_name),
    )


def create_output_filename(
    csv_path: Path | str,
    output_file_name: Optional[str] = None,
) -> str:
    path = Path(csv_path)

    if output_file_name is not None:
        filename = Path(output_file_name).name
        return filename if filename.lower().endswith(".png") else f"{filename}.png"

    tset_name, feature_name, classifier_name = infer_experiment_components(path)
    cm_name = sanitize_filename_component(path.stem)

    return (
        f"{tset_name}"
        f"__{feature_name}"
        f"__{classifier_name}"
        f"__{cm_name}.png"
    )


def get_confusion_matrix_output_path(
    csv_path: Path | str,
    output_file_name: Optional[str] = None,
    figures_root: Path | str = FIGURES_ROOT,
) -> Path:
    root = Path(figures_root)
    root.mkdir(parents=True, exist_ok=True)
    return root / create_output_filename(csv_path, output_file_name)


# Cell formatting and color scale
def format_cell_value(value: float) -> str:
    rounded_value = round(float(value))
    if np.isclose(value, rounded_value, atol=1e-10):
        return str(int(rounded_value))
    return f"{value:.2f}"


def infer_count_scale_maximum(matrix: np.ndarray) -> float:

    row_totals = np.sum(np.asarray(matrix, dtype=np.float64), axis=1)
    maximum = float(np.max(row_totals)) if row_totals.size else 0.0
    return maximum if maximum > 0.0 else 1.0


def choose_text_color(value: float, scale_maximum: float) -> str:
    threshold = scale_maximum * (2.0 / 3.0)
    return "white" if value >= threshold else "black"


# Drawing API
def draw_confusion_matrix(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    title: str,
    output_path: Path | str,
    *,
    show_figure: bool = True,
    color_scale_maximum: Optional[float] = None,
) -> Path:

    data = np.asarray(matrix, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise ValueError(f"The confusion matrix must be square: {data.shape}")
    if len(row_labels) != data.shape[0] or len(column_labels) != data.shape[1]:
        raise ValueError("Matrix labels do not match the matrix dimensions.")

    scale_maximum = (
        infer_count_scale_maximum(data)
        if color_scale_maximum is None
        else float(color_scale_maximum)
    )
    if scale_maximum <= 0.0:
        raise ValueError("color_scale_maximum must be greater than zero.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DISPLAY_DPI)
    fig.subplots_adjust(right=SUBPLOT_RIGHT)

    image = ax.imshow(
        data,
        cmap=COLOR_MAP,
        vmin=0,
        vmax=scale_maximum,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("Predicted Identity", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("True Identity", fontsize=AXIS_LABEL_FONT_SIZE)

    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(
        list(column_labels),
        fontsize=TICK_FONT_SIZE,
        rotation=X_TICK_ROTATION,
        ha="right",
    )
    ax.set_yticklabels(list(row_labels), fontsize=TICK_FONT_SIZE)

    ax.set_xticks(np.arange(-0.5, len(column_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(
        which="minor",
        linestyle=GRID_LINE_STYLE,
        linewidth=GRID_LINE_WIDTH,
        alpha=GRID_ALPHA,
    )
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_index in range(data.shape[0]):
        for column_index in range(data.shape[1]):
            value = float(data[row_index, column_index])
            if np.isclose(value, 0.0, atol=1e-12):
                continue

            cell_font_size = (
                DIAGONAL_CELL_FONT_SIZE
                if row_index == column_index
                else OFF_DIAGONAL_CELL_FONT_SIZE
            )
            ax.text(
                column_index,
                row_index,
                format_cell_value(value),
                ha="center",
                va="center",
                fontsize=cell_font_size,
                fontweight="bold",
                color=choose_text_color(value, scale_maximum),
            )

    colorbar_axis = fig.add_axes(COLORBAR_AXES)
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE)

    fig.savefig(
        destination,
        dpi=SAVE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    if not destination.exists():
        plt.close(fig)
        raise RuntimeError(
            "The confusion-matrix figure could not be saved:\n"
            f"{destination}"
        )

    if show_figure:
        plt.show()
    else:
        plt.close(fig)

    return destination


def draw_confusion_matrix_from_csv(
    csv_path: Path | str,
    *,
    title: Optional[str] = None,
    output_file_name: Optional[str] = None,
    figures_root: Path | str = FIGURES_ROOT,
    show_figure: bool = True,
    color_scale_maximum: Optional[float] = None,
) -> Path:
    path = Path(csv_path)
    matrix, row_labels, column_labels = load_confusion_matrix(path)
    figure_title = title if title is not None else create_automatic_title(path)
    output_path = get_confusion_matrix_output_path(
        path,
        output_file_name=output_file_name,
        figures_root=figures_root,
    )

    return draw_confusion_matrix(
        matrix=matrix,
        row_labels=row_labels,
        column_labels=column_labels,
        title=figure_title,
        output_path=output_path,
        show_figure=show_figure,
        color_scale_maximum=color_scale_maximum,
    )
