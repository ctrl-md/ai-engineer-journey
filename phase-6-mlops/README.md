# Phase 6 — MLOps & Production Engineering

Weeks 28–31 of the curriculum. Full plan: [`../ai_engineer_curriculum.md`](../ai_engineer_curriculum.md)

**Phase complete.**

## What's covered

- **Experiment tracking, data pipelines, dataset versioning** (Week 28): experiment tracking as the formalized version of the `performance.append({...})` pattern used throughout every prior training loop; DVC's pointer-plus-external-storage idea connected directly to CNN checkpointing (`deepcopy(cnn.state_dict())`); data pipelines as the formal version of `dataset_to_tensors`, motivated by the real `/255.0` scaling bug hit earlier in the course.
- **Model serving** (Week 29): batching as the same underlying idea as training-time batching, applied to unpredictable real-time requests instead of a fixed dataset — worked through with real numbers showing the throughput/latency tradeoff; ONNX and TensorRT; quantization's speed benefit (not just memory) traced to reduced data movement and specialized low-precision hardware.
- **Distillation, compression, software engineering for ML** (Week 30): soft labels carrying real information hard labels discard; weight decay setting up good pruning candidates as a side effect; the three ML-specific testing practices (data validation, model regression tests, reproducibility/CI-CD) that plain "does it crash" checks miss entirely.
- **Monitoring, drift detection, and the phase deliverable** (Week 31): data drift vs. concept drift distinguished precisely; drift detection framed correctly as an early-warning signal, not a replacement for eventual ground-truth accuracy — and the deliverable itself: a real FastAPI serving app (not a script that runs once), with genuine basic monitoring built and debugged by hand. Verified end to end with a real running server: a wildly out-of-distribution input correctly triggered the drift monitor while the model itself remained 99.9997% confident — a live demonstration of Week 20's calibration lesson resurfacing in a production context.

## Files

- `week_31_serve_api.py` — the serving deliverable: FastAPI `/predict`/`/metrics`/`/health` endpoints, Pydantic request validation, and `MonitoringState` (request count, rolling confidence, and a real statistical drift check) built and debugged by hand — two real bugs caught: a missing `baseline_std` assignment, and a boolean compared against a value instead of checked directly. `CNN` imported from Week 12, not re-implemented.
- `prepare_checkpoint_for_serving.py` — trains the real Week 12 CNN on real PneumoniaMNIST and bundles the trained weights with baseline input statistics into the checkpoint format the serving app expects. Run this once before starting the server.

## Status

**Phase 6 complete.** Currently on: Phase 7 — Research Skills.
