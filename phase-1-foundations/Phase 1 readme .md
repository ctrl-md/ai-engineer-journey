# Phase 1 — Math & ML Foundations

Weeks 1–4 of the curriculum. Full plan: [`../ai_engineer_curriculum.md`](../ai_engineer_curriculum.md)

## What this covers

- Vectors, matrices, dot products, matrix multiplication as a full layer
- Eigenvectors and eigenvalues
- Probability: basic probability, independent events, expectation, variance, conditional probability, Bayes' theorem
- Calculus: derivatives, the power rule, sums, partial derivatives, the chain rule, gradient descent
- Linear regression: prediction, error, loss, gradients, training loop — built and trained from scratch
- Logistic regression: sigmoid, cross-entropy loss, gradients, training loop — built and trained from scratch
- Model evaluation: overfitting/underfitting, bias/variance, train/test split, k-fold cross-validation, precision/recall, ROC-AUC

## Files

- `linear_regression.py` — predict, compute_error, compute_gradients, update_weights, train
- `logistic_regression.py` — sigmoid, predict, compute_error, compute_gradients, update_weights, train

Both verified to converge correctly: linear regression trained to an exact prediction match, logistic regression trained to over 99.9% confidence on both classes.

## Status

Complete.
