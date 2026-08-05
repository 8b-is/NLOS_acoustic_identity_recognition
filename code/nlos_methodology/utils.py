# -*- coding: utf-8 -*-
"""General filesystem, JSON, statistics, and metadata utilities."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .config import OUTPUT_ROOT, STD_DDOF
from .metadata import PUBLIC_SUBJECT_GENDER, PUBLIC_SUBJECT_ORDER


def atomic_write_text(path: Path, text: str, retries: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{int(time.time() * 1000)}.tmp"
    )
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, path)
            return
        except Exception as exc:  # pragma: no cover
            last_error = exc
            if temporary.exists():
                temporary.unlink(missing_ok=True)
            time.sleep(0.2 * (attempt + 1))
    raise OSError(f"Could not write {path}: {last_error}")


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def sample_mean_std(values: Iterable[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    mean_value = float(np.mean(array))
    std_value = float(np.std(array, ddof=STD_DDOF)) if array.size > 1 else 0.0
    return mean_value, std_value


def save_subject_metadata() -> None:
    dataframe = pd.DataFrame(
        {
            "subject": PUBLIC_SUBJECT_ORDER,
            "gender": [PUBLIC_SUBJECT_GENDER[item] for item in PUBLIC_SUBJECT_ORDER],
        }
    )
    dataframe.to_csv(
        OUTPUT_ROOT / "subject_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )
