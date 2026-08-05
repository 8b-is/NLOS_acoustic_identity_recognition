# -*- coding: utf-8 -*-

from nlos_methodology.metadata import FEATURES
from nlos_methodology.window_ablation import run_synchronized_window_ablation

ABLATION_TSET = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "C"]
ABLATION_FEATURES = FEATURES
ABLATION_MODEL_KEY = "RF"


if __name__ == "__main__":
    run_synchronized_window_ablation(
        tset=ABLATION_TSET,
        features=ABLATION_FEATURES,
        model_key=ABLATION_MODEL_KEY,
    )
