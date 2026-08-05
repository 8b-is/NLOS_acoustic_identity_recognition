# -*- coding: utf-8 -*-
"""Canonical parsing of subject-present acoustic WAV filenames."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .config import VALID_POSTURES
from .metadata import (
    ACQUISITION_SUBJECT_TO_PUBLIC,
    PUBLIC_SUBJECT_GENDER,
    SCENE_DESCRIPTIONS,
)


def parse_audio_filename(filename: str) -> Optional[Dict[str, str]]:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return None

    source = parts[0]
    surface = parts[1]
    scene = str(parts[2])
    acquisition_subject = parts[3]
    subject = ACQUISITION_SUBJECT_TO_PUBLIC.get(acquisition_subject)

    posture = "BG"
    direction = "BG"
    excitation: Optional[str] = None

    if source != "S" or subject is None:
        return None

    if len(parts) >= 7 and parts[4] in VALID_POSTURES:
        posture = parts[4]
        direction = parts[5]
        excitation = parts[6]
    elif len(parts) >= 7 and parts[4] == "Y":
        excitation = parts[5]
    elif len(parts) > 4:
        if parts[4] in VALID_POSTURES:
            posture = parts[4]
        if len(parts) > 5:
            direction = parts[5]
        excitation = parts[-1]

    if excitation is None:
        return None

    return {
        "source": source,
        "surface": surface,
        "scene": scene,
        "scene_description": SCENE_DESCRIPTIONS.get(scene, "Unknown"),
        "subject": subject,
        "gender": PUBLIC_SUBJECT_GENDER[subject],
        "posture": posture,
        "direction": direction,
        "excitation": excitation,
        "filename": filename,
    }
