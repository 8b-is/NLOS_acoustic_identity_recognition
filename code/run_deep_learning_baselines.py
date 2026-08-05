# -*- coding: utf-8 -*-

from nlos_methodology.deep_learning import (
    ALL_DEEP_MODELS,
    DeepLearningConfig,
    run_deep_learning_baselines,
)

MODELS_TO_RUN = ALL_DEEP_MODELS

CONFIG = DeepLearningConfig(
    models=MODELS_TO_RUN,
    epochs=60,
    patience=10,
    batch_size=16,
    validation_fraction=0.15,
    learning_rate=1e-3,
    weight_decay=1e-4,
    save_model_weights=False,
    force_rebuild_tensors=False,
)


if __name__ == "__main__":
    outputs = run_deep_learning_baselines(CONFIG)

    print("Deep-learning baseline analysis completed.")
    print("All reported values were calculated from model predictions.")
    print(f"Models: {', '.join(CONFIG.models)}")
    for output_name, output_path in outputs.items():
        print(f"{output_name:28s}: {output_path}")
