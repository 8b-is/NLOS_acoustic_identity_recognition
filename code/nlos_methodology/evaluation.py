# -*- coding: utf-8 -*-
"""Object-configuration LOSO evaluation for classical classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .config import (
    ENERGY_THRESHOLD_MULTIPLIER,
    LPVG_TARGET_POINTS,
    PREEMPHASIS_COEFFICIENT,
    REUSE_COMPLETED_RESULTS,
    SAMPLE_RATE,
    SECONDARY_WINDOW_SIZE,
    STD_DDOF,
    STFT_EXPECTED_BINS,
    STFT_HOP_LENGTH,
    STFT_N_FFT,
    STFT_WIN_LENGTH,
)
from .metadata import (
    MODEL_SUBJECT_ORDER,
    PUBLICATION_CLASS_IDS,
    PUBLIC_SUBJECT_GENDER,
    PUBLIC_SUBJECT_ORDER,
    SCENE_DESCRIPTIONS,
    decode_subject_ids,
    encode_subject_labels,
    manuscript_tset_label,
    public_tset_members,
    subject_genders,
    tset_storage_tag,
)
from .protocols import iter_object_configuration_loso
from .preprocessing import (
    INDEPENDENT_WINDOW_STRATEGY,
    normalize_window_strategy,
    window_strategy_label,
    window_strategy_storage_tag,
)
from .reporting import save_classification_report_csv, save_confusion_matrix_csv
from .utils import atomic_write_text, sample_mean_std, write_json


def run_loso_experiment(
    features: np.ndarray,
    subject_labels: np.ndarray,
    gender_labels: np.ndarray,
    scenes: np.ndarray,
    surfaces: np.ndarray,
    tset: Sequence[str],
    feature_name: str,
    model_key: str,
    model_label: str,
    base_estimator,
    output_directory: Path,
    window_strategy: str = INDEPENDENT_WINDOW_STRATEGY,
) -> Optional[Dict[str, object]]:
    strategy = normalize_window_strategy(window_strategy)
    summary_path = output_directory / "LOSO_summary.csv"
    done_path = output_directory / "DONE.txt"

    if REUSE_COMPLETED_RESULTS and summary_path.exists() and done_path.exists():
        existing = pd.read_csv(summary_path)
        if len(existing) == 1:
            print(
                f"[RESULT] Reused completed result: {model_label}, "
                f"T-set={manuscript_tset_label(tset)}, feature={feature_name}, "
                f"window={window_strategy_label(strategy)}"
            )
            reused = existing.iloc[0].to_dict()
            reused.setdefault("window_strategy", strategy)
            reused.setdefault(
                "window_strategy_label",
                window_strategy_label(strategy),
            )
            return reused

    output_directory.mkdir(parents=True, exist_ok=True)

    expected_genders = subject_genders(subject_labels)
    if not np.array_equal(np.asarray(gender_labels, dtype=object), expected_genders):
        raise ValueError("Subject and gender labels are inconsistent.")

    encoded_labels = encode_subject_labels(subject_labels)
    class_names = list(PUBLIC_SUBJECT_ORDER)
    class_ids = PUBLICATION_CLASS_IDS.copy()
    unique_scenes = sorted(pd.unique(scenes).tolist())

    fold_rows: List[Dict[str, object]] = []
    fold_confusion_matrices: List[np.ndarray] = []

    for fold in iter_object_configuration_loso(scenes, surfaces):
        held_out_scene = fold.held_out_scene
        train_mask = fold.train_mask
        test_mask = fold.test_mask

        if not np.any(train_mask) or not np.any(test_mask):
            print(
                f"[LOSO] Scene {held_out_scene}: insufficient train/test data; "
                "fold skipped."
            )
            continue

        x_train = features[train_mask]
        x_test = features[test_mask]
        y_train = encoded_labels[train_mask]
        y_test = encoded_labels[test_mask]

        estimator = clone(base_estimator)
        estimator.fit(x_train, y_train)
        y_pred = estimator.predict(x_test)

        accuracy = float(accuracy_score(y_test, y_pred))
        precision, recall, f1_macro, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        fold_rows.append(
            {
                "classifier_key": model_key,
                "classifier": model_label,
                "tset_tag": tset_storage_tag(tset),
                "tset": manuscript_tset_label(tset),
                "feature": feature_name,
                "window_strategy": strategy,
                "window_strategy_label": window_strategy_label(strategy),
                "fold": fold.fold_index,
                "held_out_scene": str(held_out_scene),
                "scene_description": SCENE_DESCRIPTIONS.get(
                    str(held_out_scene), "Unknown"
                ),
                "accuracy": accuracy,
                "accuracy_percent": accuracy * 100.0,
                "precision_macro": float(precision),
                "recall_macro": float(recall),
                "f1_macro": float(f1_macro),
                "n_train": int(x_train.shape[0]),
                "n_test": int(x_test.shape[0]),
            }
        )

        matrix = confusion_matrix(y_test, y_pred, labels=class_ids)
        fold_confusion_matrices.append(matrix)

        fold_prefix = f"LOSO_scene_{held_out_scene}_fold_{fold.fold_index:02d}"
        save_confusion_matrix_csv(
            matrix,
            class_names,
            output_directory / f"{fold_prefix}_CM.csv",
        )
        save_classification_report_csv(
            y_test,
            y_pred,
            class_ids,
            class_names,
            output_directory / f"{fold_prefix}_classification_report.csv",
        )

        true_subjects = decode_subject_ids(y_test)
        predicted_subjects = decode_subject_ids(y_pred)
        pd.DataFrame(
            {
                "true_subject": true_subjects,
                "predicted_subject": predicted_subjects,
                "true_gender": subject_genders(true_subjects),
                "predicted_gender": subject_genders(predicted_subjects),
                "held_out_scene": str(held_out_scene),
            }
        ).to_csv(
            output_directory / f"{fold_prefix}_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"[LOSO] {model_label} | T-set={manuscript_tset_label(tset)} | "
            f"feature={feature_name} | "
            f"window={window_strategy_storage_tag(strategy)} | "
            f"scene={held_out_scene} | "
            f"accuracy={accuracy * 100.0:.2f}%"
        )

    if not fold_rows:
        print(f"[LOSO] No valid folds for {model_label}, {feature_name}.")
        return None

    fold_dataframe = pd.DataFrame(fold_rows)
    fold_dataframe.to_csv(
        output_directory / "LOSO_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    accuracy_mean, accuracy_std = sample_mean_std(fold_dataframe["accuracy"])
    precision_mean, precision_std = sample_mean_std(
        fold_dataframe["precision_macro"]
    )
    recall_mean, recall_std = sample_mean_std(fold_dataframe["recall_macro"])
    f1_mean, f1_std = sample_mean_std(fold_dataframe["f1_macro"])

    summary_row: Dict[str, object] = {
        "classifier_key": model_key,
        "classifier": model_label,
        "tset_tag": tset_storage_tag(tset),
        "tset": manuscript_tset_label(tset),
        "feature": feature_name,
        "window_strategy": strategy,
        "window_strategy_label": window_strategy_label(strategy),
        "validation": "Object-configuration-based LOSO",
        "n_folds": int(len(fold_dataframe)),
        "std_ddof": STD_DDOF,
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "accuracy_mean_percent": accuracy_mean * 100.0,
        "accuracy_std_percent": accuracy_std * 100.0,
        "precision_macro_mean": precision_mean,
        "precision_macro_std": precision_std,
        "recall_macro_mean": recall_mean,
        "recall_macro_std": recall_std,
        "f1_macro_mean": f1_mean,
        "f1_macro_std": f1_std,
        "n_samples": int(len(features)),
        "n_male_subjects": int(
            sum(PUBLIC_SUBJECT_GENDER[item] == "Male" for item in class_names)
        ),
        "n_female_subjects": int(
            sum(PUBLIC_SUBJECT_GENDER[item] == "Female" for item in class_names)
        ),
        "feature_dimension": int(features.shape[1]),
    }

    pd.DataFrame([summary_row]).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    overall_matrix = np.sum(fold_confusion_matrices, axis=0)
    save_confusion_matrix_csv(
        overall_matrix,
        class_names,
        output_directory / "LOSO_overall_CM.csv",
    )

    write_json(
        output_directory / "experiment_metadata.json",
        {
            "classifier_key": model_key,
            "classifier": model_label,
            "tset_tag": tset_storage_tag(tset),
            "tset": manuscript_tset_label(tset),
            "tset_members": public_tset_members(tset),
            "feature": feature_name,
            "window_strategy": strategy,
            "window_strategy_label": window_strategy_label(strategy),
            "validation": "Object-configuration-based LOSO",
            "class_names": class_names,
            "subject_genders": {
                subject: PUBLIC_SUBJECT_GENDER[subject] for subject in class_names
            },
            "model_encoding_order": MODEL_SUBJECT_ORDER,
            "scenes": [str(scene) for scene in unique_scenes],
            "sample_rate": SAMPLE_RATE,
            "secondary_window_size": SECONDARY_WINDOW_SIZE,
            "energy_threshold_multiplier": ENERGY_THRESHOLD_MULTIPLIER,
            "preemphasis_coefficient": PREEMPHASIS_COEFFICIENT,
            "stft_n_fft": STFT_N_FFT,
            "stft_hop_length": STFT_HOP_LENGTH,
            "stft_win_length": STFT_WIN_LENGTH,
            "stft_bins": STFT_EXPECTED_BINS,
            "lpvg_target_points": LPVG_TARGET_POINTS,
            "std_ddof": STD_DDOF,
        },
    )
    atomic_write_text(done_path, "DONE\n")

    print(
        f"[LOSO] Completed {model_label} | "
        f"T-set={manuscript_tset_label(tset)} | feature={feature_name} | "
        f"window={window_strategy_storage_tag(strategy)} | "
        f"accuracy={accuracy_mean * 100.0:.2f} ± "
        f"{accuracy_std * 100.0:.2f}%"
    )
    return summary_row
