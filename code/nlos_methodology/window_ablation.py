# -*- coding: utf-8 -*-
"""Independent-versus-synchronized secondary-window ablation analysis.

The ablation keeps every component fixed except the secondary-echo window
selection strategy:

- independent: each microphone selects its own final above-threshold window;
- synchronized: the eight channel energies are averaged at each time index,
  one common final above-threshold index is selected, and the same boundaries
  are applied to all microphones.

"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import ABLATION_ROOT, ROOT_AUDIO_FOLDER, ensure_output_directories
from .dataset import RawDataTuple, prepare_raw_segments
from .evaluation import run_loso_experiment
from .feature_cache import get_or_build_features
from .metadata import (
    FEATURES,
    SUBJECTS_BY_SURFACE,
    manuscript_tset_label,
    tset_storage_tag,
)
from .models import MODELS
from .preprocessing import (
    INDEPENDENT_WINDOW_STRATEGY,
    SYNCHRONIZED_WINDOW_STRATEGY,
    window_strategy_label,
    window_strategy_storage_tag,
)
from .utils import write_json


WINDOW_ABLATION_NAME = "Synchronized_Window"
DEFAULT_WINDOW_STRATEGIES = (
    INDEPENDENT_WINDOW_STRATEGY,
    SYNCHRONIZED_WINDOW_STRATEGY,
)


def window_ablation_root(tset: Sequence[str]) -> Path:
    return (
        ABLATION_ROOT
        / WINDOW_ABLATION_NAME
        / f"TSET_{tset_storage_tag(tset)}"
    )


def _assert_paired_datasets(
    independent: RawDataTuple,
    synchronized: RawDataTuple,
) -> None:
    independent_raw, independent_subjects, independent_genders, independent_scenes, independent_surfaces = independent
    synchronized_raw, synchronized_subjects, synchronized_genders, synchronized_scenes, synchronized_surfaces = synchronized

    if independent_raw.shape != synchronized_raw.shape:
        raise RuntimeError(
            "Independent and synchronized raw datasets are not paired: "
            f"independent shape={independent_raw.shape}, "
            f"synchronized shape={synchronized_raw.shape}."
        )

    comparisons = {
        "subjects": (independent_subjects, synchronized_subjects),
        "genders": (independent_genders, synchronized_genders),
        "scenes": (independent_scenes, synchronized_scenes),
        "surfaces": (independent_surfaces, synchronized_surfaces),
    }
    for name, (left, right) in comparisons.items():
        if not np.array_equal(left, right):
            raise RuntimeError(
                "Independent and synchronized datasets do not have the same "
                f"sample ordering for {name}. Delete the synchronized-window "
                "cache and rebuild both branches from the same audio root."
            )


def _valid_subject_mask(subjects: np.ndarray) -> np.ndarray:
    valid_subjects = SUBJECTS_BY_SURFACE["D"]
    return np.asarray(
        [str(subject) in valid_subjects for subject in subjects],
        dtype=bool,
    )


def _strategy_output_directory(
    tset: Sequence[str],
    window_strategy: str,
    feature_name: str,
    model_key: str,
) -> Path:
    return (
        window_ablation_root(tset)
        / window_strategy_storage_tag(window_strategy)
        / feature_name
        / model_key
    )


def _load_fold_metrics(
    output_directory: Path,
    window_strategy: str,
) -> pd.DataFrame:
    path = output_directory / "LOSO_fold_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Fold metrics were not created: {path}")
    dataframe = pd.read_csv(path)
    keep = [
        "feature",
        "fold",
        "held_out_scene",
        "scene_description",
        "accuracy",
        "accuracy_percent",
        "n_train",
        "n_test",
    ]
    dataframe = dataframe[keep].copy()
    prefix = window_strategy_storage_tag(window_strategy).lower()
    rename = {
        "accuracy": f"{prefix}_accuracy",
        "accuracy_percent": f"{prefix}_accuracy_percent",
        "n_train": f"{prefix}_n_train",
        "n_test": f"{prefix}_n_test",
    }
    return dataframe.rename(columns=rename)


def _build_summary_table(
    summary_rows: List[Dict[str, object]],
    feature_order: Sequence[str],
) -> pd.DataFrame:
    summary = pd.DataFrame(summary_rows)
    required = {
        "feature",
        "window_strategy",
        "accuracy_mean_percent",
        "accuracy_std_percent",
        "n_folds",
        "n_samples",
        "feature_dimension",
    }
    missing = required - set(summary.columns)
    if missing:
        raise RuntimeError(f"Ablation summaries are missing columns: {sorted(missing)}")

    rows: List[Dict[str, object]] = []
    for feature_name in feature_order:
        independent = summary[
            (summary["feature"] == feature_name)
            & (summary["window_strategy"] == INDEPENDENT_WINDOW_STRATEGY)
        ]
        synchronized = summary[
            (summary["feature"] == feature_name)
            & (summary["window_strategy"] == SYNCHRONIZED_WINDOW_STRATEGY)
        ]
        if len(independent) != 1 or len(synchronized) != 1:
            raise RuntimeError(
                f"Expected one result per strategy for feature {feature_name}."
            )

        ind = independent.iloc[0]
        sync = synchronized.iloc[0]
        ind_mean = float(ind["accuracy_mean_percent"])
        ind_std = float(ind["accuracy_std_percent"])
        sync_mean = float(sync["accuracy_mean_percent"])
        sync_std = float(sync["accuracy_std_percent"])

        rows.append(
            {
                "feature": feature_name,
                "independent_mean_accuracy_percent": ind_mean,
                "independent_std_percent": ind_std,
                "independent_mean_plus_minus_std": f"{ind_mean:.2f} ± {ind_std:.2f}",
                "synchronized_mean_accuracy_percent": sync_mean,
                "synchronized_std_percent": sync_std,
                "synchronized_mean_plus_minus_std": f"{sync_mean:.2f} ± {sync_std:.2f}",
                "synchronized_minus_independent_pp": sync_mean - ind_mean,
                "n_folds": int(ind["n_folds"]),
                "n_samples": int(ind["n_samples"]),
                "feature_dimension": int(ind["feature_dimension"]),
            }
        )
    return pd.DataFrame(rows)


def _build_paired_fold_table(
    tset: Sequence[str],
    features: Sequence[str],
    model_key: str,
) -> pd.DataFrame:
    paired_frames: List[pd.DataFrame] = []
    merge_keys = [
        "feature",
        "fold",
        "held_out_scene",
        "scene_description",
    ]

    for feature_name in features:
        independent_directory = _strategy_output_directory(
            tset,
            INDEPENDENT_WINDOW_STRATEGY,
            feature_name,
            model_key,
        )
        synchronized_directory = _strategy_output_directory(
            tset,
            SYNCHRONIZED_WINDOW_STRATEGY,
            feature_name,
            model_key,
        )
        independent = _load_fold_metrics(
            independent_directory,
            INDEPENDENT_WINDOW_STRATEGY,
        )
        synchronized = _load_fold_metrics(
            synchronized_directory,
            SYNCHRONIZED_WINDOW_STRATEGY,
        )
        paired = independent.merge(
            synchronized,
            on=merge_keys,
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(independent) or len(paired) != len(synchronized):
            raise RuntimeError(
                f"Fold pairing failed for feature {feature_name}."
            )
        paired["synchronized_minus_independent_pp"] = (
            paired["synchronized_accuracy_percent"]
            - paired["independent_accuracy_percent"]
        )
        paired_frames.append(paired)

    return pd.concat(paired_frames, ignore_index=True)


def run_synchronized_window_ablation(
    tset: Sequence[str],
    features: Sequence[str] = FEATURES,
    model_key: str = "RF",
    audio_root: Path = ROOT_AUDIO_FOLDER,
) -> pd.DataFrame:
    ensure_output_directories()

    if model_key not in MODELS:
        raise KeyError(f"Unknown model key: {model_key}")
    model_specification = MODELS[model_key]
    model_label = str(model_specification["label"])

    invalid_features = [item for item in features if item not in FEATURES]
    if invalid_features:
        raise ValueError(f"Unknown feature representations: {invalid_features}")

    print("Synchronized-window ablation analysis")
    print(f"Audio root : {audio_root}")
    print(f"Output root: {window_ablation_root(tset)}")
    print(f"T-set      : {manuscript_tset_label(tset)}")
    print(f"Features   : {list(features)}")
    print(f"Classifier : {model_label}")


    independent_data = prepare_raw_segments(
        audio_root,
        tset,
        window_strategy=INDEPENDENT_WINDOW_STRATEGY,
    )
    synchronized_data = prepare_raw_segments(
        audio_root,
        tset,
        window_strategy=SYNCHRONIZED_WINDOW_STRATEGY,
    )
    _assert_paired_datasets(independent_data, synchronized_data)

    _, subjects, genders, scenes, surfaces = independent_data
    valid_mask = _valid_subject_mask(subjects)
    if not np.any(valid_mask):
        raise RuntimeError("No valid wall-surface subjects were found.")

    subject_subset = subjects[valid_mask]
    gender_subset = genders[valid_mask]
    scene_subset = scenes[valid_mask]
    surface_subset = surfaces[valid_mask]

    raw_by_strategy = {
        INDEPENDENT_WINDOW_STRATEGY: independent_data[0],
        SYNCHRONIZED_WINDOW_STRATEGY: synchronized_data[0],
    }
    summary_rows: List[Dict[str, object]] = []

    for window_strategy in DEFAULT_WINDOW_STRATEGIES:
        raw_segments = raw_by_strategy[window_strategy]
        print(f"Window strategy: {window_strategy_label(window_strategy)}")


        for feature_name in features:
            print(
                f"Feature={feature_name} | "
                f"window={window_strategy_storage_tag(window_strategy)}"
            )

            full_feature_matrix = get_or_build_features(
                raw_segments,
                tset,
                feature_name,
                window_strategy=window_strategy,
            )
            feature_subset = full_feature_matrix[valid_mask]
            output_directory = _strategy_output_directory(
                tset,
                window_strategy,
                feature_name,
                model_key,
            )

            summary = run_loso_experiment(
                features=feature_subset,
                subject_labels=subject_subset,
                gender_labels=gender_subset,
                scenes=scene_subset,
                surfaces=surface_subset,
                tset=tset,
                feature_name=feature_name,
                model_key=model_key,
                model_label=model_label,
                base_estimator=model_specification["estimator"],
                output_directory=output_directory,
                window_strategy=window_strategy,
            )
            if summary is None:
                raise RuntimeError(
                    f"No valid LOSO result for {feature_name}, {window_strategy}."
                )
            summary_rows.append(summary)

    experiment_root = window_ablation_root(tset)
    summary_table = _build_summary_table(summary_rows, features)
    summary_path = experiment_root / "Synchronized_Window_Ablation_summary.csv"
    summary_table.to_csv(summary_path, index=False, encoding="utf-8-sig")

    paired_folds = _build_paired_fold_table(tset, features, model_key)
    paired_path = experiment_root / "Synchronized_Window_Ablation_fold_metrics.csv"
    paired_folds.to_csv(paired_path, index=False, encoding="utf-8-sig")

    write_json(
        experiment_root / "Synchronized_Window_Ablation_metadata.json",
        {
            "analysis": "Independent versus synchronized secondary-window ablation",
            "tset_tag": tset_storage_tag(tset),
            "tset": manuscript_tset_label(tset),
            "features": list(features),
            "classifier_key": model_key,
            "classifier": model_label,
            "validation": "Object-configuration-based LOSO",
            "independent_method": (
                "Adaptive threshold and final above-threshold window selected "
                "separately for each microphone channel."
            ),
            "synchronized_method": (
                "Window energies at each time index averaged across eight "
                "microphones; adaptive threshold computed from the common "
                "sequence; final above-threshold index applied to all channels."
            ),
            "paired_sample_count": int(np.count_nonzero(valid_mask)),
            "summary_csv": str(summary_path),
            "paired_fold_csv": str(paired_path),
        },
    )

    print("Synchronized-window ablation completed.")
    print(f"Summary     : {summary_path}")
    print(f"Paired folds: {paired_path}")
    return summary_table
