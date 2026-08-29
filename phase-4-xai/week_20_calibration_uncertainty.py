"""
Week 20 -- Calibration and uncertainty quantification; counterfactual
explanations; light look at adversarial robustness.

Calibration: a reliability check -- bucket predictions by their
stated confidence, then check what fraction were actually correct.

MC Dropout: keep dropout ACTIVE at inference time (the opposite of
the normal model.eval() practice), run the same input through the
model multiple times, and use the SPREAD across those runs as an
uncertainty signal -- low spread means a robust prediction that
doesn't depend on which neurons dropout happened to disable; high
spread means a fragile, uncertain one.

Counterfactual explanations and adversarial robustness were covered
conceptually, not built here:
- Counterfactuals: "what's the smallest input change that flips the
  prediction" -- essentially distance and direction from a decision
  boundary (Phase 1), made concrete and patient-specific.
- Adversarial robustness: gradient ASCENT on the INPUT (not descent
  on the weights) -- the same gradient computation used throughout
  this curriculum, just pointed at making the loss worse instead of
  better, and applied to pixels instead of parameters. The trained
  model itself is never touched, only what gets fed into it.
"""

import torch


def calibration_check(confidences, correct):
    """
    confidences: list of stated confidence values (e.g. all ~0.9)
    correct:     list of booleans, whether each prediction was actually right
    Returns the stated average confidence vs the actual accuracy.
    """
    avg_confidence = sum(confidences) / len(confidences)
    actual_accuracy = sum(correct) / len(correct)
    return avg_confidence, actual_accuracy


def mc_dropout_predict(model, x, num_samples):
    """
    Runs the same input through the model num_samples times with
    dropout ACTIVE (model.train() mode), returning all predictions
    so their spread can be examined.
    """
    model.train()  # deliberately keep dropout active
    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            output = model(x)
            predictions.append(output.argmax(dim=-1))
    return predictions


if __name__ == "__main__":
    # calibration worked example: 20 predictions, all ~90% confidence
    confidences = [0.90] * 20
    correct = [True] * 12 + [False] * 8  # only 12/20 actually correct
    avg_conf, actual_acc = calibration_check(confidences, correct)
    print(f"stated confidence: {avg_conf:.2f}, actual accuracy: {actual_acc:.2f}")
    print(f"overconfident by: {avg_conf - actual_acc:.2f}")
