# -*- coding: utf-8 -*-
"""Dataset labels, subject metadata, scene names, and T-set definitions."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


ACQUISITION_SUBJECT_TO_PUBLIC = {
    "C1": "S1",
    "C2": "S2",
    "C3": "S3",
    "C4": "S4",
    "C6": "S5",
    "C7": "S6",
    "C8": "S7",
    "C9": "S8",
    "C10": "S9",
    "C11": "S10",
}

PUBLIC_SUBJECT_ORDER = [
    "S1", "S2", "S3", "S4", "S5",
    "S6", "S7", "S8", "S9", "S10",
]

PUBLIC_SUBJECT_GENDER = {
    "S1": "Male",
    "S2": "Male",
    "S3": "Male",
    "S4": "Male",
    "S5": "Male",
    "S6": "Female",
    "S7": "Female",
    "S8": "Female",
    "S9": "Female",
    "S10": "Female",
}

SUBJECTS_BY_SURFACE = {
    "D": set(PUBLIC_SUBJECT_ORDER),
}

MODEL_SUBJECT_ORDER = [
    "S1", "S9", "S10", "S2", "S3",
    "S4", "S5", "S6", "S7", "S8",
]
SUBJECT_TO_MODEL_ID = {
    subject: class_id for class_id, subject in enumerate(MODEL_SUBJECT_ORDER)
}
MODEL_ID_TO_SUBJECT = {
    class_id: subject for subject, class_id in SUBJECT_TO_MODEL_ID.items()
}
PUBLICATION_CLASS_IDS = np.asarray(
    [SUBJECT_TO_MODEL_ID[subject] for subject in PUBLIC_SUBJECT_ORDER],
    dtype=int,
)

SCENE_DESCRIPTIONS = {
    "0": "No object",
    "1": "Wood",
    "2": "Iron",
    "3": "Brick",
    "4": "White appliance",
    "5": "PVC Panel",
    "6": "White appliance and PVC",
    "7": "Wood and PVC",
    "8": "Brick and wood",
    "9": "Iron and brick",
}

DATASET_MULTI_TRANSMITTER_TOKEN = "C"
PUBLIC_MULTI_TRANSMITTER_LABEL = "M"

T_SETS: List[List[str]] = [
    ["T1"], ["T2"], ["T3"], ["T4"], ["T5"], ["T6"], ["T7"], ["T8"], ["C"],
    ["T6", "T7"], ["T6", "C"], ["T7", "C"],
    ["T1", "T2"], ["T3", "T4"], ["T5", "T6"], ["T7", "T8"],
    ["T1", "T2", "T3", "T4"],
    ["T5", "T6", "T7", "T8"],
    ["T1", "T3", "T5", "T7"],
    ["T2", "T4", "T6", "T8"],
    ["T6", "T7", "C"],
    ["T1", "T2", "T3", "T4", "T5"],
    ["T4", "T5", "T6", "T7", "T8"],
    ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"],
    ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "C"],
]

FEATURES = [
    "STFT",
    "LPVG_D0",
    "LPVG_D1",
    "LPVG_D2",
    "Fusion_D0",
    "Fusion_D1",
    "Fusion_D2",
]


def public_excitation_label(token: str) -> str:
    """Convert the acquisition token C to the manuscript label M."""
    if token == DATASET_MULTI_TRANSMITTER_TOKEN:
        return PUBLIC_MULTI_TRANSMITTER_LABEL
    return token


def tset_storage_tag(tset: Sequence[str]) -> str:
    return "_".join(public_excitation_label(token) for token in tset)


def manuscript_tset_label(tset: Sequence[str]) -> str:
    return "+".join(public_excitation_label(token) for token in tset)


def public_tset_members(tset: Sequence[str]) -> List[str]:
    return [public_excitation_label(token) for token in tset]


def encode_subject_labels(subject_labels: Sequence[str]) -> np.ndarray:
    unknown = sorted(set(subject_labels) - set(SUBJECT_TO_MODEL_ID))
    if unknown:
        raise ValueError(f"Unknown publication subject labels: {unknown}")
    return np.asarray(
        [SUBJECT_TO_MODEL_ID[str(subject)] for subject in subject_labels],
        dtype=int,
    )


def decode_subject_ids(class_ids: Sequence[int]) -> np.ndarray:
    return np.asarray(
        [MODEL_ID_TO_SUBJECT[int(class_id)] for class_id in class_ids],
        dtype=object,
    )


def subject_genders(subject_labels: Sequence[str]) -> np.ndarray:
    unknown = sorted(set(subject_labels) - set(PUBLIC_SUBJECT_GENDER))
    if unknown:
        raise ValueError(f"Gender is unavailable for subjects: {unknown}")
    return np.asarray(
        [PUBLIC_SUBJECT_GENDER[str(subject)] for subject in subject_labels],
        dtype=object,
    )
