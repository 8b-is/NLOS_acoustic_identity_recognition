# -*- coding: utf-8 -*-
"""Classical classifier definitions"""

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import GLOBAL_SEED


MODELS = {
    "RF": {
        "label": "Random Forest",
        "estimator": RandomForestClassifier(
            n_estimators=200,
            random_state=GLOBAL_SEED,
            n_jobs=-1,
        ),
    },
    "ET": {
        "label": "Extra Trees",
        "estimator": ExtraTreesClassifier(
            n_estimators=300,
            random_state=GLOBAL_SEED,
            n_jobs=-1,
        ),
    },
    "SVM": {
        "label": "SGD-based Linear SVM",
        "estimator": make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            SGDClassifier(
                loss="hinge",
                alpha=1e-4,
                penalty="l2",
                class_weight="balanced",
                random_state=GLOBAL_SEED,
                max_iter=5_000,
                tol=1e-3,
            ),
        ),
    },
}
