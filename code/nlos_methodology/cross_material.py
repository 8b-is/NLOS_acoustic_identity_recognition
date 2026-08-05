# -*- coding: utf-8 -*-
"""Cross-material results reported in the revised manuscript.

This module uses the shared NLOS methodology modules for raw-segment
construction, feature extraction, subject encoding, and model definition.
No accuracy, standard deviation, LOMO result, or transfer-matrix value is
entered manually.

Computed experiments
--------------------
1. Single-Source Transfer:
   20 directed source-material -> target-material experiments.
2. Leave-One-Material-Out (LOMO):
   five folds, each trained on four materials and tested on the unseen fifth.
3. Representations:
   STFT, LPVG (d=0), and Fusion (d=0).
4. Classifier:
   the shared 200-tree Random Forest definition, MODELS["RF"].
5. T-set:
   T1-T8 plus the simultaneous multi-transmitter condition C (reported as M).

"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from . import dataset as dataset_module
from .config import OUTPUT_ROOT, ROOT_AUDIO_FOLDER, ensure_output_directories
from .feature_cache import get_or_build_features
from .metadata import (
    PUBLIC_SUBJECT_ORDER,
    encode_subject_labels,
    manuscript_tset_label,
    tset_storage_tag,
)
from .models import MODELS


# Fixed experiment
TSET: Tuple[str, ...] = (
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "C"
)

FEATURES: Tuple[str, ...] = (
    "STFT",
    "LPVG_D0",
    "Fusion_D0",
)

FEATURE_DISPLAY_NAMES: Mapping[str, str] = {
    "STFT": "STFT",
    "LPVG_D0": "LPVG (d=0)",
    "Fusion_D0": "Fusion",
}

MODEL_KEY = "RF"
WINDOW_STRATEGY = "independent"
CSV_SEPARATOR = ";"

SINGLE_MATERIAL_SCENES: Mapping[str, str] = {
    "1": "Wood",
    "2": "Iron",
    "3": "Brick",
    "4": "White appliance",
    "5": "PVC",
}
MATERIAL_ORDER: Tuple[str, ...] = tuple(SINGLE_MATERIAL_SCENES.keys())

RESULT_DIRECTORY = (
    OUTPUT_ROOT
    / "cross_material"
    / f"TSET_{tset_storage_tag(TSET)}"
)
FIGURE_DIRECTORY = (
    OUTPUT_ROOT
    / "Figures"
    / "Cross_Material"
    / f"TSET_{tset_storage_tag(TSET)}"
)


# Output helpers
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


def _sample_mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]

    if array.size == 0:
        return float("nan"), float("nan")

    mean_value = float(np.mean(array))
    standard_deviation = (
        float(np.std(array, ddof=1))
        if array.size > 1
        else 0.0
    )
    return mean_value, standard_deviation


# Shared data and feature access
def _load_raw_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load shared raw segments and return raw, subject, and scene arrays."""

    if hasattr(dataset_module, "prepare_raw_dataset"):
        bundle = dataset_module.prepare_raw_dataset(
            ROOT_AUDIO_FOLDER,
            TSET,
            window_strategy=WINDOW_STRATEGY,
        )
        return (
            np.asarray(bundle.raw_segments),
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

    raw_segments = np.asarray(output[0])
    subjects = np.asarray(output[1], dtype=str)
    scenes = np.asarray(output[3], dtype=str)

    return raw_segments, subjects, scenes


def _get_feature_matrix(
    raw_segments: np.ndarray,
    feature_name: str,
) -> np.ndarray:
    try:
        matrix = get_or_build_features(
            raw_segments,
            TSET,
            feature_name,
            window_strategy=WINDOW_STRATEGY,
        )
    except TypeError:
        matrix = get_or_build_features(
            raw_segments,
            TSET,
            feature_name,
        )

    matrix = np.asarray(matrix, dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError(
            f"Feature matrix must be two-dimensional: {feature_name}, "
            f"shape={matrix.shape}."
        )

    if matrix.shape[0] != len(raw_segments):
        raise ValueError(
            f"Feature-row mismatch for {feature_name}: "
            f"{matrix.shape[0]} != {len(raw_segments)}."
        )

    if not np.isfinite(matrix).all():
        raise ValueError(
            f"Feature matrix contains NaN or Inf: {feature_name}."
        )

    return matrix


# Experimental-design validation
def _validate_material_dataset(
    subjects: np.ndarray,
    scenes: np.ndarray,
) -> None:
    """Verify the 120-sample and 10-identity design for Scenes 1-5."""

    expected_subjects = set(PUBLIC_SUBJECT_ORDER)

    for scene_code in MATERIAL_ORDER:
        scene_mask = scenes == scene_code
        scene_subjects = subjects[scene_mask]
        subject_counts = pd.Series(scene_subjects).value_counts()

        if int(np.sum(scene_mask)) != 120:
            raise ValueError(
                f"Scene {scene_code} ({SINGLE_MATERIAL_SCENES[scene_code]}) "
                f"contains {int(np.sum(scene_mask))} samples; expected 120."
            )

        if set(scene_subjects) != expected_subjects:
            raise ValueError(
                f"Scene {scene_code} does not contain exactly S1-S10."
            )

        if (
            len(subject_counts) != 10
            or not bool((subject_counts.to_numpy(dtype=int) == 12).all())
        ):
            raise ValueError(
                f"Scene {scene_code} must contain 12 samples per subject."
            )


def _validate_train_test_classes(
    y_train: np.ndarray,
    y_test: np.ndarray,
    context: str,
) -> None:
    training_classes = set(np.asarray(y_train).tolist())
    test_classes = set(np.asarray(y_test).tolist())
    missing_classes = sorted(test_classes - training_classes)

    if missing_classes:
        raise ValueError(
            f"{context}: target identities absent from training: "
            f"{missing_classes}."
        )

# Model execution

def _fit_predict_accuracy(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    context: str,
) -> float:
    if not np.any(train_mask):
        raise ValueError(f"{context}: empty training set.")

    if not np.any(test_mask):
        raise ValueError(f"{context}: empty test set.")

    y_train = labels[train_mask]
    y_test = labels[test_mask]
    _validate_train_test_classes(y_train, y_test, context)

    estimator = clone(MODELS[MODEL_KEY]["estimator"])
    estimator.fit(feature_matrix[train_mask], y_train)
    predictions = estimator.predict(feature_matrix[test_mask])

    return float(accuracy_score(y_test, predictions) * 100.0)


def _run_single_source_transfer(
    feature_matrices: Mapping[str, np.ndarray],
    labels: np.ndarray,
    scenes: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for feature_name in FEATURES:
        feature_matrix = feature_matrices[feature_name]

        for source_scene in MATERIAL_ORDER:
            for target_scene in MATERIAL_ORDER:
                if source_scene == target_scene:
                    continue

                train_mask = scenes == source_scene
                test_mask = scenes == target_scene
                context = (
                    f"{SINGLE_MATERIAL_SCENES[source_scene]} -> "
                    f"{SINGLE_MATERIAL_SCENES[target_scene]}"
                )

                accuracy_percent = _fit_predict_accuracy(
                    feature_matrix,
                    labels,
                    train_mask,
                    test_mask,
                    context,
                )

                rows.append(
                    {
                        "model": MODEL_KEY,
                        "feature": feature_name,
                        "feature_representation": (
                            FEATURE_DISPLAY_NAMES[feature_name]
                        ),
                        "source_scene": source_scene,
                        "source_material": (
                            SINGLE_MATERIAL_SCENES[source_scene]
                        ),
                        "target_scene": target_scene,
                        "target_material": (
                            SINGLE_MATERIAL_SCENES[target_scene]
                        ),
                        "n_train": int(np.sum(train_mask)),
                        "n_test": int(np.sum(test_mask)),
                        "accuracy_percent": accuracy_percent,
                    }
                )

    results = pd.DataFrame(rows)

    expected_rows = len(FEATURES) * 20
    if len(results) != expected_rows:
        raise RuntimeError(
            f"Unexpected Single-Source result count: {len(results)}; "
            f"expected {expected_rows}."
        )

    return results


def _run_lomo(
    feature_matrices: Mapping[str, np.ndarray],
    labels: np.ndarray,
    scenes: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    material_scene_array = np.asarray(MATERIAL_ORDER, dtype=str)

    for feature_name in FEATURES:
        feature_matrix = feature_matrices[feature_name]

        for target_scene in MATERIAL_ORDER:
            train_scenes = tuple(
                scene
                for scene in MATERIAL_ORDER
                if scene != target_scene
            )
            train_mask = np.isin(scenes, material_scene_array[material_scene_array != target_scene])
            test_mask = scenes == target_scene
            context = (
                "LOMO target="
                f"{SINGLE_MATERIAL_SCENES[target_scene]}"
            )

            accuracy_percent = _fit_predict_accuracy(
                feature_matrix,
                labels,
                train_mask,
                test_mask,
                context,
            )

            rows.append(
                {
                    "model": MODEL_KEY,
                    "feature": feature_name,
                    "feature_representation": (
                        FEATURE_DISPLAY_NAMES[feature_name]
                    ),
                    "target_scene": target_scene,
                    "target_material": (
                        SINGLE_MATERIAL_SCENES[target_scene]
                    ),
                    "train_scenes": "+".join(train_scenes),
                    "n_train": int(np.sum(train_mask)),
                    "n_test": int(np.sum(test_mask)),
                    "accuracy_percent": accuracy_percent,
                }
            )

    results = pd.DataFrame(rows)

    expected_rows = len(FEATURES) * 5
    if len(results) != expected_rows:
        raise RuntimeError(
            f"Unexpected LOMO result count: {len(results)}; "
            f"expected {expected_rows}."
        )

    return results

# table and transfer matrices
def _build_table(
    single_source_results: pd.DataFrame,
    lomo_results: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for feature_name in FEATURES:
        single_values = single_source_results.loc[
            single_source_results["feature"] == feature_name,
            "accuracy_percent",
        ].to_numpy(dtype=float)

        lomo_values = lomo_results.loc[
            lomo_results["feature"] == feature_name,
            "accuracy_percent",
        ].to_numpy(dtype=float)

        single_mean, single_sd = _sample_mean_std(single_values)
        lomo_mean, lomo_sd = _sample_mean_std(lomo_values)

        rows.append(
            {
                "Feature Representation": (
                    FEATURE_DISPLAY_NAMES[feature_name]
                ),
                "Single Source Transfer ACC (%)": single_mean,
                "Single Source Transfer SD (%)": single_sd,
                "Single Source Transfer ACC ± SD (%)": (
                    f"{single_mean:.2f} ± {single_sd:.2f}"
                ),
                "LOMO ACC (%)": lomo_mean,
                "LOMO SD (%)": lomo_sd,
                "LOMO ACC ± SD (%)": (
                    f"{lomo_mean:.2f} ± {lomo_sd:.2f}"
                ),
            }
        )

    return pd.DataFrame(rows)


def _build_transfer_matrices(
    single_source_results: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    matrices: Dict[str, np.ndarray] = {}

    for feature_name in FEATURES:
        matrix = np.full((5, 5), np.nan, dtype=np.float64)
        subset = single_source_results[
            single_source_results["feature"] == feature_name
        ]

        for _, result_row in subset.iterrows():
            source_index = MATERIAL_ORDER.index(
                str(result_row["source_scene"])
            )
            target_index = MATERIAL_ORDER.index(
                str(result_row["target_scene"])
            )
            matrix[source_index, target_index] = float(
                result_row["accuracy_percent"]
            )

        if np.isfinite(np.diag(matrix)).any():
            raise RuntimeError(
                f"Diagonal values must remain blank: {feature_name}."
            )

        if int(np.sum(np.isfinite(matrix))) != 20:
            raise RuntimeError(
                f"Transfer matrix must contain 20 values: {feature_name}."
            )

        matrices[feature_name] = matrix

    return matrices



# Figures
PALETTE = ["#16324F", "#2A628F", "#4EA5D9", "#7FC8A9", "#F6D55C"]
TRANSFER_CMAP = LinearSegmentedColormap.from_list(
    "cross_material_manuscript",
    PALETTE,
    N=256,
)
TRANSFER_CMAP.set_bad(color="#ECECEC")


def _draw_figure(
    matrices: Mapping[str, np.ndarray],
    output_path: Path,
) -> None:
    material_labels = [
        SINGLE_MATERIAL_SCENES[scene]
        for scene in MATERIAL_ORDER
    ]
    panel_order = (
        ("LPVG_D0", "LPVG (d=0)"),
        ("STFT", "STFT"),
        ("Fusion_D0", "Fusion (d=0)"),
    )

    normalization = Normalize(vmin=0.0, vmax=100.0)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16.2, 5.7),
        dpi=220,
        constrained_layout=True,
    )

    image = None

    for panel_index, (feature_name, panel_title) in enumerate(panel_order):
        axis = axes[panel_index]
        values = np.asarray(matrices[feature_name], dtype=np.float64)
        masked_values = np.ma.masked_invalid(values)

        image = axis.imshow(
            masked_values,
            cmap=TRANSFER_CMAP,
            norm=normalization,
            interpolation="nearest",
            aspect="equal",
        )

        axis.set_title(
            panel_title,
            fontsize=15,
            fontweight="normal",
            pad=12,
        )
        axis.set_xlabel("Test material", fontsize=15, labelpad=7)

        if panel_index == 0:
            axis.set_ylabel(
                "Training material",
                fontsize=15,
                labelpad=8,
            )

        axis.set_xticks(np.arange(len(material_labels)))
        axis.set_yticks(np.arange(len(material_labels)))
        axis.set_xticklabels(
            material_labels,
            rotation=35,
            ha="right",
            fontsize=11,
        )
        axis.set_yticklabels(
            material_labels
            if panel_index == 0
            else [""] * len(material_labels),
            fontsize=11,
        )

        axis.set_xticks(
            np.arange(-0.5, len(material_labels), 1),
            minor=True,
        )
        axis.set_yticks(
            np.arange(-0.5, len(material_labels), 1),
            minor=True,
        )
        axis.grid(which="minor", color="white", linewidth=1.2)
        axis.tick_params(which="minor", bottom=False, left=False)

        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]

                if not np.isfinite(value):
                    continue

                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color="black",
                )

        for spine in axis.spines.values():
            spine.set_visible(False)

    if image is None:
        raise RuntimeError("Figure could not be created.")

    colorbar = figure.colorbar(
        image,
        ax=axes,
        fraction=0.025,
        pad=0.025,
    )
    colorbar.set_label(
        "Identity accuracy (%)",
        fontsize=13,
        labelpad=10,
    )
    colorbar.ax.tick_params(labelsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


# Public entry point

def run_cross_material_analysis() -> Dict[str, Path]:

    ensure_output_directories()
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    raw_segments, subjects, scenes = _load_raw_data()

    if not (
        len(raw_segments)
        == len(subjects)
        == len(scenes)
    ):
        raise ValueError(
            "Raw segments, subject labels, and scene labels are not aligned."
        )

    _validate_material_dataset(subjects, scenes)

    labels = encode_subject_labels(subjects)
    feature_matrices = {
        feature_name: _get_feature_matrix(
            raw_segments,
            feature_name,
        )
        for feature_name in FEATURES
    }

    single_source_results = _run_single_source_transfer(
        feature_matrices,
        labels,
        scenes,
    )
    lomo_results = _run_lomo(
        feature_matrices,
        labels,
        scenes,
    )
    table_1 = _build_table(
        single_source_results,
        lomo_results,
    )
    transfer_matrices = _build_transfer_matrices(
        single_source_results
    )

    single_source_path = (
        RESULT_DIRECTORY
        / "single_source_transfer_results.csv"
    )
    lomo_path = RESULT_DIRECTORY / "LOMO_results.csv"
    table_1_path = (
        RESULT_DIRECTORY
        / "table_1_cross_material_summary.csv"
    )

    _save_csv(single_source_results, single_source_path)
    _save_csv(lomo_results, lomo_path)
    _save_csv(table_1, table_1_path)

    matrix_paths: Dict[str, Path] = {}
    material_labels = [
        SINGLE_MATERIAL_SCENES[scene]
        for scene in MATERIAL_ORDER
    ]

    for feature_name, matrix in transfer_matrices.items():
        matrix_dataframe = pd.DataFrame(
            matrix,
            index=material_labels,
            columns=material_labels,
        )
        matrix_path = (
            RESULT_DIRECTORY
            / f"single_source_transfer_matrix_{feature_name}.csv"
        )
        _save_csv(
            matrix_dataframe,
            matrix_path,
            index=True,
        )
        matrix_paths[feature_name] = matrix_path

    figure_1_path = (
        FIGURE_DIRECTORY
        / "Figure_1_cross_material_transfer_matrices.png"
    )
    _draw_figure(
        transfer_matrices,
        figure_1_path,
    )

    outputs: Dict[str, Path] = {
        "single_source_results": single_source_path,
        "lomo_results": lomo_path,
        "table_1": table_1_path,
        "figure_1": figure_1_path,
    }
    outputs.update(
        {
            f"matrix_{feature_name}": matrix_path
            for feature_name, matrix_path in matrix_paths.items()
        }
    )

    print(f"T-set          : {manuscript_tset_label(TSET)}")
    print(f"Classifier     : {MODELS[MODEL_KEY]['label']}")
    print(f"RF model fits  : {len(FEATURES) * (20 + 5)}")
    print("Manual results : none")

    return outputs
