# Phase 7 — Research Skills

Reading a paper efficiently (abstract → figures → method → results) and reproducing one at small scale.

## Weeks 32–34: Paper reproduction

Reproduced Maheshwari et al., _"Feature Engineering Combined with 1D Convolutional Neural Network for Improved Mortality Prediction,"_ on the PhysioNet/CinC 2012 Challenge dataset — 4,000 ICU patients, first 48 hours of vitals and labs, predicting in-hospital mortality.

Built end to end from raw files: a parser for PhysioNet's irregular-timestamp format (including catching missing-value sentinels and physically implausible readings), feature engineering (mean/min/max/missingness-flag per clinical variable), population-mean imputation, a 1D-CNN matching the paper's architecture, training with their exponential learning-rate decay, and evaluation with best-checkpoint selection on a held-out validation set.

**Result**: 0.82 test AUC vs. the paper's best of 0.848. Deviations from the paper — the feature statistics used, the imputation strategy, no domain-engineered features — are documented in `REPORT.md`.

Run it: `python weeks_33_34_mortality_cnn_reproduction.py` — downloads the PhysioNet data itself on first run.
