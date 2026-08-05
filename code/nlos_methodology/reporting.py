# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from .metadata import PUBLIC_SUBJECT_GENDER


def save_confusion_matrix_csv(
    matrix: np.ndarray,
    class_names: Sequence[str],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(
        np.asarray(matrix, dtype=np.int64),
        index=list(class_names),
        columns=list(class_names),
    )
    dataframe.index.name = "true_subject"
    dataframe.columns.name = "predicted_subject"
    dataframe.to_csv(
        path,
        index=True,
        index_label="true_subject",
        encoding="utf-8-sig",
    )


def save_classification_report_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_ids: np.ndarray,
    class_names: Sequence[str],
    path: Path,
) -> None:
    report = classification_report(
        y_true,
        y_pred,
        labels=class_ids,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    dataframe = pd.DataFrame(report).transpose()
    dataframe.insert(
        0,
        "gender",
        [PUBLIC_SUBJECT_GENDER.get(str(index), "") for index in dataframe.index],
    )
    dataframe.index.name = "subject"
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=True, encoding="utf-8-sig")
