# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

GLOBAL_SEED = 42
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

ROOT_AUDIO_FOLDER = Path(
    os.environ.get(
        "NLOS_AUDIO_ROOT",
        r"D:\Tubitak 123E386_Ses_Lazer_Data\Ham Data\Ses\Duvar",
    )
)
OUTPUT_ROOT = Path(
    os.environ.get(
        "NLOS_OUTPUT_ROOT",
        "NLOS_identity_recognition_results",
    )
)
CACHE_ROOT = OUTPUT_ROOT / "cache"
RESULTS_ROOT = OUTPUT_ROOT / "results"
FIGURES_ROOT = OUTPUT_ROOT / "Figures"
ABLATION_ROOT = OUTPUT_ROOT / "ablation"
CACHE_SCHEMA_VERSION = "nlos-v1"
REUSE_COMPLETED_RESULTS = True


# Signal processing and feature extraction
SAMPLE_RATE = 44_100
SECONDARY_WINDOW_SIZE = 1_024
ENERGY_THRESHOLD_MULTIPLIER = 1.0
PREEMPHASIS_COEFFICIENT = 0.97
N_MICROPHONES = 8

STFT_N_FFT = 512
STFT_HOP_LENGTH = 256
STFT_WIN_LENGTH = 512
STFT_EXPECTED_BINS = STFT_N_FFT // 2 + 1

LPVG_TARGET_POINTS = 128
LPVG_DEGREES = (0, 1, 2)

N_JOBS_FEATURE = max(1, (os.cpu_count() or 2) - 1)
PARALLEL_BACKEND = "loky"
PARALLEL_VERBOSE = 5
PARALLEL_BATCH_SIZE = 1

STD_DDOF = 1
VALID_POSTURES = {"A", "O", "C"}
ALLOWED_SURFACES = {"D"}


def ensure_output_directories() -> None:
    """Create the release-owned output, cache, and results directories."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
