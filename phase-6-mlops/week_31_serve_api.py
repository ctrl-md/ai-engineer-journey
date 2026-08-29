"""
Week 31 -- Deploy a served API with monitoring. The Phase 6 deliverable.

A FastAPI service wrapping a trained CNN: a /predict endpoint that
loads the model ONCE at startup (not per-request -- the whole point
of serving vs. a one-off script), and a /metrics endpoint exposing
real, basic production monitoring:
- request count and average confidence (a proxy for whether the
  model's certainty is drifting over time)
- a genuine drift check -- comparing each incoming input's pixel
  mean against a baseline mean/std computed from training data,
  flagging inputs that fall more than 2 standard deviations away
  (Week 31's "data drift" concept, actually implemented, not just
  described)

Pydantic's request/response models (PredictRequest/PredictResponse)
are real data validation, for free -- Week 30's "data validation"
concept: a malformed request (wrong type, missing field) gets
rejected automatically, before it ever reaches the model.

CNN is imported from Week 12, not re-implemented.
"""

import os
import sys
import time
from typing import List

import torch
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "phase-2-deep-learning-core"
    ),
)
from week_12_capstone_pneumonia_cnn import CNN


class MonitoringState:
    """Basic production monitoring: request volume, rolling average
    confidence, and a simple statistical drift check against a stored
    training-data baseline."""

    def __init__(self, baseline_mean, baseline_std, drift_threshold_stds=2.0) -> None:
        self.drift_threshold_stds = drift_threshold_stds
        self.baseline_mean = baseline_mean
        self.baseline_std = baseline_std
        self.request_count = 0
        self.confidence_sum = 0.0
        self.drift_flags = 0

    def record(self, confidence, input_mean):
        self.request_count += 1
        self.confidence_sum += confidence
        baseline_mean_drift = (
            abs(input_mean - self.baseline_mean)
            > self.drift_threshold_stds * self.baseline_std
        )
        if baseline_mean_drift:
            self.drift_flags += 1

    def summary(self):
        zero_check = self.request_count if self.request_count != 0 else 1
        average_confidence = self.confidence_sum / zero_check
        drift_flag_rate = self.drift_flags / zero_check
        return {
            "request_count": self.request_count,
            "average_confidence": average_confidence,
            "drift_flags": self.drift_flags,
            "drift_flag_rate": drift_flag_rate,
        }


class PredictRequest(BaseModel):
    pixels: List[float]  # flattened 28x28 image, 784 values


class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    latency_ms: float


app = FastAPI(title="Pneumonia Classifier Serving API")

checkpoint = torch.load(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_checkpoint.pt"),
    weights_only=False,
)
cnn = CNN(
    in_channels=1,
    hidden_channels=12,
    kernel_size=3,
    padding=1,
    dropout_rate=0.5,
    num_classes=2,
)
cnn.load_state_dict(checkpoint["model_state"])
cnn.eval()  # loaded ONCE at startup -- never re-loaded per request

monitor = MonitoringState(
    baseline_mean=checkpoint["baseline_mean"],
    baseline_std=checkpoint["baseline_std"],
)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start = time.time()
    x = torch.tensor(request.pixels, dtype=torch.float32).reshape(1, 1, 28, 28)

    with torch.no_grad():
        logits = cnn(x)
        probs = torch.softmax(logits, dim=1)
        pred = int(probs.argmax(dim=1).item())
        confidence = probs[0, pred].item()

    latency_ms = (time.time() - start) * 1000
    monitor.record(confidence, x.mean().item())

    return PredictResponse(
        prediction=pred, confidence=confidence, latency_ms=latency_ms
    )


@app.get("/metrics")
def metrics():
    return monitor.summary()


@app.get("/health")
def health():
    return {"status": "ok"}
