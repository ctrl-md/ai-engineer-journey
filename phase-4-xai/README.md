# Phase 4 — Explainable AI (XAI)

Weeks 18–21 of the curriculum. Full plan: [`../ai_engineer_curriculum.md`](../ai_engineer_curriculum.md)

**Phase complete.**

## What's covered

- **Feature attribution** (Week 18): LIME and SHAP covered conceptually (local perturbation-based vs. game-theoretic, and why SHAP's exact computation is exponential in the number of features); Integrated Gradients built from scratch and verified against a hand-calculated linear model.
- **Visual explanation** (Week 19): saliency maps (the single-gradient method IG improves on), Grad-CAM (the full method worked by hand — gradients per channel, averaged into importance weights, weighted sum of spatially-structured feature maps — then coded and verified against the hand calculation), attention visualization (already-computed, no extra work, but genuinely multi-headed).
- **Calibration, uncertainty, counterfactuals, adversarial robustness** (Week 20): a reliability-bucket check for overconfidence, MC Dropout (deliberately keeping dropout active to use its randomness as an uncertainty signal), counterfactuals as distance-and-direction from a decision boundary, and adversarial robustness as gradient ascent on the input instead of descent on the weights.
- **Fairness auditing and the phase deliverable** (Week 21): subgroup performance disparities, the demographic-parity-vs-equalized-odds tension, and the deliverable itself — Integrated Gradients applied to a real, properly-trained CNN prediction (cleanly localized attribution, 15.2 vs ~2 across quadrants) and attention visualization applied to a real GPT prediction (correctly concentrated on the word being completed), tied together in a written interpretation report.

## Files

- `week-18-feature-attribution.py` — `integrated_gradients`, verified against a hand-calculated linear model
- `week-19-visual-explanation.py` — Grad-CAM, coded and verified against the hand-worked example
- `week-20-calibration-uncertainty.py` — calibration reliability check, MC Dropout
- `week-21-cnn-integrated-gradients.py` — IG applied to a real, properly-trained pneumonia classifier
- `week-21-gpt-attention-viz.py` — real attention weights extracted from a trained GPT
- `week-21-interpretation-report.md` — the written deliverable, tying both applications together

## Status

**Phase 4 complete.** Currently on: Phase 5 — Applied Generative AI & NLP, starting with pretraining vs. fine-tuning vs. RLHF/DPO.
