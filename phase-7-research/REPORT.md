# Reproducing a 1D-CNN ICU Mortality Prediction Paper

Part of a self-taught AI engineering curriculum — the "read a paper, reproduce the core result" phase. No tutorial for this one, just the paper and PhysioNet.

**Repo**: https://github.com/ctrl-md/ai-engineer-journey/tree/main/phase-7-research

## The paper

Maheshwari et al., _"Feature Engineering Combined with 1D Convolutional Neural Network for Improved Mortality Prediction."_ They took the PhysioNet/CinC 2012 Challenge dataset — 4,000 ICU patients, first 48 hours of vitals and labs — and built a 1D CNN to predict who dies in-hospital. Best result: **0.848 AUC**.

## What I actually built

Everything, from raw files up:

- **Parser** for PhysioNet's raw format — irregular timestamps, missing-value sentinels (`-1`), and yes, a real patient record with a pH of `733.0` that had to get caught and dropped
- **Feature engineering** — mean, min, max, and a "was this ever measured" flag per clinical variable. The paper doesn't say exactly what stat they used, so I made my own call here and documented it
- **Population-mean imputation** for the ~82% missing-data rate this dataset is known for
- **The CNN itself** — 4 conv layers (32/32/64/64), batchnorm, dropout, matching their architecture as closely as the paper's description allowed
- **Training** — Adam, their exponential LR decay formula, proper train/val/test split, best-checkpoint selection on validation loss (not just whatever the last epoch happened to land on)

No scikit-learn, no copy-pasted Kaggle notebook. Every function above got written, tested, and debugged from a blank file.

## Where I deviated from the paper, on purpose

Papers leave things out. That's normal, not a flaw — but it means a "reproduction" always involves real decisions the paper didn't make for you. Mine:

- Mean/min/max/flag instead of whatever single stat they used
- My own missing-data handling, including catching the `-1` sentinel and a couple of physically impossible values in the raw data
- Best-checkpoint-by-validation-loss instead of reporting the final epoch

None of this is hidden. It's why the number below isn't identical to theirs.

## Results

| Model                              | AUC       |
| ---------------------------------- | --------- |
| SAPS-I (clinical scoring baseline) | 0.313     |
| SVM                                | 0.791     |
| KNN                                | 0.799     |
| Random Forest                      | 0.822     |
| XGBoost                            | 0.840     |
| **My reproduction (1D-CNN)**       | **0.820** |
| Paper's 1D-CNN (best)              | 0.848     |

Core claim holds: CNN beats the classic baselines and the clinical scoring system by a wide margin. `0.028` off their best number, with a different feature scheme and no domain-engineered features (they added a BUN/Creatinine ratio and a clinical sum-of-labs term I didn't bother with).

## Running it

```
cd phase-7-research
python weeks_33_34_mortality_cnn_reproduction.py
```

Downloads the PhysioNet data itself on first run (no credentialing needed for this specific dataset — open access), then parses, trains, and evaluates end to end.

## Why this, why now

Aiming for AI + medicine long-term. This is the "can you actually read a paper and rebuild what it claims, not just call an API" checkpoint — one piece of a longer curriculum, not the whole story.
