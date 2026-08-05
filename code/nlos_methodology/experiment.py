# -*- coding: utf-8 -*-
"""Classical RF/ET/SGD-based SVM experiments."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .config import OUTPUT_ROOT, RESULTS_ROOT, ROOT_AUDIO_FOLDER, ensure_output_directories
from .dataset import prepare_raw_segments
from .evaluation import run_loso_experiment
from .feature_cache import get_or_build_features
from .metadata import (
    FEATURES,
    SUBJECTS_BY_SURFACE,
    T_SETS,
    manuscript_tset_label,
    tset_storage_tag,
)
from .models import MODELS
from .utils import save_subject_metadata


def save_global_summary(rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    dataframe = pd.DataFrame(rows)
    sort_columns = ["tset_tag", "feature", "classifier_key"]
    dataframe = dataframe.sort_values(sort_columns, kind="stable")
    dataframe.to_csv(
        OUTPUT_ROOT / "GLOBAL_LOSO_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    ensure_output_directories()
    save_subject_metadata()

    print("NLOS identity-recognition manuscript evaluation pipeline")
    print(f"Audio root : {ROOT_AUDIO_FOLDER}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"T-sets     : {len(T_SETS)}")
    print(f"Features   : {len(FEATURES)}")
    print(f"Classifiers: {len(MODELS)}")
    print("Confusion matrices will be saved as CSV files.")
    print("All caches will be created under the current output root.")


    global_summary_rows: List[Dict[str, object]] = []

    for tset_index, tset in enumerate(T_SETS, start=1):
        tset_tag = tset_storage_tag(tset)
        print(
            f"T-set {tset_index}/{len(T_SETS)}: "
            f"{manuscript_tset_label(tset)} (storage tag: {tset_tag})"
        )

        raw_segments, subjects, genders, scenes, surfaces = prepare_raw_segments(
            ROOT_AUDIO_FOLDER,
            tset,
        )
        if len(raw_segments) == 0:
            print(f"[DATA] No valid samples for T-set {tset_tag}; skipped.")
            continue

        valid_subjects = SUBJECTS_BY_SURFACE["D"]
        valid_mask = np.asarray(
            [subject in valid_subjects for subject in subjects],
            dtype=bool,
        )
        if not np.any(valid_mask):
            print(f"[DATA] No valid wall-surface subjects for T-set {tset_tag}.")
            continue

        subject_subset = subjects[valid_mask]
        gender_subset = genders[valid_mask]
        scene_subset = scenes[valid_mask]
        surface_subset = surfaces[valid_mask]

        for feature_name in FEATURES:
            print(
                f"Feature: {feature_name} | "
                f"T-set: {manuscript_tset_label(tset)}"
            )

            full_feature_matrix = get_or_build_features(
                raw_segments,
                tset,
                feature_name,
            )
            feature_subset = full_feature_matrix[valid_mask]

            for model_key, model_specification in MODELS.items():
                model_label = str(model_specification["label"])
                output_directory = (
                    RESULTS_ROOT
                    / f"TSET_{tset_tag}"
                    / feature_name
                    / model_key
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
                )
                if summary is not None:
                    global_summary_rows.append(summary)
                    save_global_summary(global_summary_rows)

    save_global_summary(global_summary_rows)
    print("All configured experiments are complete.")
    print(f"Global summary: {OUTPUT_ROOT / 'GLOBAL_LOSO_summary.csv'}")
