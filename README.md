# NLOS_acoustic_identity_recognition

**Zenodo DOI:** https://doi.org/10.5281/zenodo.21802627 

## Topological–Spectral Fusion of Secondary Acoustic Echoes for Identity Recognition in Non-Line-of-Sight Environments

This repository contains the Python source code, detailed dataset metadata,
representative anonymized multichannel acoustic recordings, and reproducibility
documentation associated with the manuscript:

**Topological–Spectral Fusion of Secondary Acoustic Echoes for Identity Recognition in Non-Line-of-Sight Environments**

## Study Overview

The study investigates person identity recognition in a controlled airborne
non-line-of-sight (NLOS) environment using delayed multichannel acoustic
responses. Measurements were acquired from 10 anonymous subjects with an active
array consisting of eight speakers and eight synchronized microphones. The
experimental design includes 10 object configurations, three postures, four
orientations, eight sequential transmitter conditions, and one simultaneous
multi-transmitter condition.

Raw recordings are divided into non-overlapping 1,024-sample windows. The final
window exceeding an adaptive recording-specific energy threshold is selected as
the secondary-echo segment. Time-averaged STFT features and LPVG degree-sequence
features are extracted and combined through feature-level concatenation.

The repository includes the final implementations for:

- Random Forest, Extra Trees, and SGD-based linear SVM experiments;
- object-configuration-based leave-one-scene-out (LOSO) validation;
- independent and synchronized secondary-window selection ablation;
- deep-learning baselines using raw and STFT-based inputs;
- single-source cross-material transfer and leave-one-material-out analyses;
- matched human-present versus human-empty validation. 

## Software Requirements

Install the core dependencies with:

```bash
pip install -r requirements.txt
```

The workflow was developed in Python and uses packages including NumPy, pandas, SciPy, librosa, scikit-learn, matplotlib, NetworkX, joblib, and PyTorch for the deep-learning baselines. A suitable PyTorch build should be installed for the target CPU or GPU environment.

## How to Reproduce the Analysis

The scripts are designed to be opened in Spyder and executed with **F5**. They may also be run from a terminal.

Set the full-data location before running an analysis:

```text
NLOS_AUDIO_ROOT = path to the wall-reflection WAV dataset
NLOS_OUTPUT_ROOT = optional output directory
```

Main entry points:

```text
code/run_classical_loso.py
code/run_synchronized_window_ablation.py
code/run_deep_learning_baselines.py
code/run_cross_material_analysis.py
code/run_human_empty_validation.py
```

Command-line example:

```bash
cd code
python run_classical_loso.py
```

## Citation

Please cite the associated article and Zenodo record when using this repository. The citation metadata are also provided in `CITATION.cff`.

```text
Olgun, N. (2026). NLOS Acoustic Identity Recognition: Source Code and
Reproducibility Materials, version 1.0.0. Zenodo.
https://doi.org/10.5281/zenodo.21802627
```
