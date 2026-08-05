# NLOS identity-recognition methodology

Environment variables remain unchanged:

- `NLOS_AUDIO_ROOT`: root folder containing the wall-reflection WAV files.
- `NLOS_OUTPUT_ROOT`: output root for caches and results.

## Modules

- `config.py`: fixed sampling, secondary-window, STFT, LPVG, parallel, cache,
  and reproducibility settings.
- `metadata.py`: S1-S10 mapping, gender labels, class encoding, scene names,
  T-sets, and feature names.
- `parsing.py`: canonical WAV filename parser.
- `preprocessing.py`: independent secondary-echo selection,  synchronized secondary-echo selection, pre-emphasis, and
  per-segment peak normalization.
- `features.py`: STFT, LPVG d=0/1/2, fusion, and feature-matrix construction.
- `dataset.py`: WAV discovery, sample assembly, and raw-segment cache.
- `feature_cache.py`: feature cache creation and validation.
- `protocols.py`: reusable object-configuration LOSO split masks.
- `models.py`: RF, ET, and SGD-based linear SVM definitions.
- `reporting.py`: confusion-matrix and classification-report CSV writers.
- `evaluation.py`: classical object-configuration LOSO evaluation.
- `experiment.py`: orchestration of T-set × feature × classifier experiments.
- `visualization.py`: reusable publication-style confusion-matrix loading, naming, drawing, and saving under `OUTPUT_ROOT/Figures`.

