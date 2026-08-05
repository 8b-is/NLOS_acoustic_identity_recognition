# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ObjectConfigurationLOSOFold:
    fold_index: int
    held_out_scene: str
    train_mask: np.ndarray
    test_mask: np.ndarray


def iter_object_configuration_loso(
    scenes: np.ndarray,
    surfaces: np.ndarray,
    target_surface: str = "D",
) -> Iterator[ObjectConfigurationLOSOFold]:
    unique_scenes = sorted(pd.unique(scenes).tolist())
    for fold_index, held_out_scene in enumerate(unique_scenes, start=1):
        test_mask = (scenes == held_out_scene) & (surfaces == target_surface)
        train_mask = (scenes != held_out_scene) & (surfaces == target_surface)
        yield ObjectConfigurationLOSOFold(
            fold_index=fold_index,
            held_out_scene=str(held_out_scene),
            train_mask=train_mask,
            test_mask=test_mask,
        )
