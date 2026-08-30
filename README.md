# AI Engineer Journey

A structured, 39-week self-study path from deep learning foundations to applied generative AI/NLP and explainable AI — aimed at becoming an AI engineer, not just someone who calls fine-tuning APIs.

## About this repo

Final-year medical student moving into AI engineering, working toward a career at the intersection of AI and medicine. This repo tracks that journey — every exercise, every from-scratch implementation, every project — one hour a day.

The guiding principle behind the whole curriculum: build the core mechanism from scratch once, before reaching for the library that does it for you.

## The plan

- Full curriculum: [`ai_engineer_curriculum.md`](./ai_engineer_curriculum.md)
- Progress tracker: [`PROGRESS.md`](./PROGRESS.md)

8 phases, 39 weeks:

1. Math & ML Foundations (Weeks 1–4)
2. Deep Learning Core (Weeks 5–12)
3. Transformers & Modern Architectures (Weeks 13–17)
4. Explainable AI (Weeks 18–21)
5. Applied Generative AI & NLP (Weeks 22–27)
6. MLOps & Production Engineering (Weeks 28–31)
7. Research Skills (Weeks 32–34)
8. Capstone: AI + Medicine (Weeks 35–39)

## Structure

Each phase has its own folder, with a subfolder per week. Deliverables — the real, working code — live in the week they belong to.

## Status

**Phases 1–7 complete (Weeks 1–34)** — math foundations through linear/logistic regression; a from-scratch autograd engine, NumPy MLP, and PyTorch fundamentals; CNNs and RNNs, capped by a CNN trained on a medical imaging dataset with checkpointing; attention and a GPT built and trained from scratch; explainability techniques (SHAP, LIME, Grad-CAM, calibration, fairness auditing) applied to earlier models; LoRA fine-tuning, RAG, and a clinical Q&A tool; a served, monitored API with drift detection; and a full paper reproduction — a 1D-CNN for ICU mortality prediction on the PhysioNet 2012 Challenge dataset, built end to end from raw files to a trained model, reaching 0.82 test AUC against the paper's 0.848. See each phase's folder for details.

Currently on: Phase 8 — Capstone: AI + Medicine.
