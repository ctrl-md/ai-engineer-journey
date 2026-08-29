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

**Phase 1 complete** — math foundations (vectors, matrices, eigenvectors, probability through Bayes' theorem, calculus through gradient descent), plus linear regression and logistic regression built, trained, and evaluated from scratch. See `phase-1-foundations/`.

**Phase 2 complete** — forward pass and backpropagation by hand, a working autograd engine from scratch, a batched multi-layer perceptron in NumPy, PyTorch fundamentals, normalization/regularization, CNNs with a working ResNet-style residual block, RNNs/LSTMs/GRUs and GPU/systems basics, and a full capstone project training a CNN on real medical imaging data (PneumoniaMNIST) with experiment tracking, checkpointing, and real debugging along the way. See `phase-2-deep-learning-core/`.

**Phase 3 complete** — attention built from scratch and verified against hand calculations, a full transformer block, a complete GPT trained from scratch with causal masking rigorously verified, real BPE tokenization, MLM/CLM and encoder/decoder reasoning, vision transformers, a conceptual pass on VAEs/GANs/diffusion models, and the phase deliverable: dropout added, trained on a richer corpus, and genuine autoregressive text generation working end to end. See `phase-3-transformers/`.

**Phase 4 complete** — Integrated Gradients built from scratch and verified, Grad-CAM worked by hand and coded, calibration and MC Dropout for uncertainty, counterfactuals and adversarial robustness reasoned through, fairness auditing (including the demographic-parity-vs-equalized-odds tension), and the phase deliverable: real XAI applied to the actual trained CNN and GPT, with a written interpretation report. See `phase-4-xai/`.

Currently on: Phase 5 — Applied Generative AI & NLP, starting with pretraining vs. fine-tuning vs. RLHF/DPO.
