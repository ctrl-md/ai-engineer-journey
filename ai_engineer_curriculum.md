# AI Engineer Curriculum — Deep Learning, XAI, Applied GenAI/NLP

### For a future in AI + Medicine

## Guiding principle

Every phase below follows the same rule: **build the core mechanism from scratch once, before you touch the library that does it for you.** Anyone can call `model.fit()` or hit a fine-tuning API. What makes you an engineer instead of a user of engineering is that when something breaks, or behaves strangely, or needs to be adapted to a problem no tutorial covered, you know what's actually happening underneath and can reason about it — or rebuild it.

That's the difference this curriculum is designed around.

---

## Phase 1 — Math & ML Foundations

_Weeks 1–4_

- **Linear algebra**: vectors, matrices, matrix multiplication, dot products, eigenvalues/eigenvectors, SVD (used constantly in dimensionality reduction and understanding embeddings)
- **Calculus**: derivatives, partial derivatives, the chain rule, gradients, Jacobians — the machinery backprop is built on
- **Probability & statistics**: distributions, Bayes' theorem, MLE/MAP estimation, expectation/variance, basics of hypothesis testing
- **Optimization**: gradient descent and its variants (SGD, momentum, Adam), why convexity matters (and why deep nets mostly aren't convex)
- **Classical ML**: linear/logistic regression, decision trees, ensembles — still the right tool for a lot of tabular clinical data. Bias-variance tradeoff, cross-validation, and evaluation metrics (precision/recall/ROC-AUC — these matter enormously in clinical model evaluation, where false negatives and false positives carry very different costs)

**Deliverable**: implement linear regression and logistic regression from scratch in NumPy — no scikit-learn — so gradient descent stops being an abstraction.

---

## Phase 2 — Deep Learning Core

_Weeks 5–12_

- Forward pass, backpropagation derivation, computational graphs
- **Build a tiny autograd engine from scratch** (a "micrograd"-style exercise) — this single exercise does more for real understanding of backprop than any amount of reading
- Implement a multi-layer perceptron from scratch in NumPy _before_ touching PyTorch
- Then: PyTorch fundamentals — tensors, autograd, `nn.Module`, training loops, `DataLoader`
- Weight initialization, batch norm / layer norm, dropout, weight decay, learning rate scheduling
- **CNNs**: convolution and pooling mechanics, classic architectures (ResNet family) — directly relevant to medical imaging
- **RNNs/LSTMs/GRUs**: sequence modeling, the vanishing gradient problem
- **GPU/systems basics**: how memory, mixed precision (fp16/bf16), and hardware use affect training speed — the systems layer that decides whether your model trains in an hour or a week
- Debugging deep nets: reading loss curves, gradient checking, diagnosing overfitting vs underfitting

**Deliverable**: train a CNN on a medical imaging dataset (e.g. chest X-ray classification) from scratch, with proper experiment tracking.

---

## Phase 3 — Transformers & Modern Architectures

_Weeks 13–17_

- Derive and implement scaled dot-product attention from scratch
- Full transformer architecture: multi-head attention, positional encoding, layer norm placement, feedforward blocks
- **Build a small GPT-style transformer from scratch and train it on a toy corpus** — the single exercise that demystifies every LLM you'll ever use after this
- Tokenization: BPE, WordPiece, SentencePiece
- Pretraining objectives: masked language modeling vs causal language modeling
- Encoder vs decoder vs encoder-decoder — when each is the right choice
- Vision transformers (ViT) — increasingly used in medical imaging alongside CNNs
- **Other core generative architectures**: VAEs, GANs, and diffusion models — you don't need to master these, just know how they differ from transformers and when each is used. Diffusion especially shows up a lot in medical image synthesis and augmentation.

**Deliverable**: a working small GPT trained from scratch (nanoGPT-style) on a text corpus of your choice.

---

## Phase 4 — Explainable AI (XAI)

_Weeks 18–21_

- Feature attribution: SHAP, LIME, integrated gradients
- Visual explanation: Grad-CAM, saliency maps, attention visualization — critical for imaging models
- Counterfactual explanations
- Model-intrinsic vs post-hoc interpretability
- **Calibration and uncertainty quantification** — a clinical model needs to know when it doesn't know
- Fairness and bias auditing in clinical models
- A light look at adversarial robustness for imaging models — how small, deliberately crafted changes to an image can fool a classifier, and why that matters when a model's safety failure has real consequences
- A light touch of mechanistic interpretability (the current research frontier for understanding what's actually happening inside a network)

**Deliverable**: apply XAI techniques to your Phase 2 imaging model and Phase 3 language model. Write it up as if you were explaining the model's reasoning to a clinician who has to trust — or challenge — its output.

---

## Phase 5 — Applied Generative AI & NLP

_Weeks 22–27_

This is the phase that separates "calls a fine-tuning API" from "understands what fine-tuning is doing."

- Pretraining vs fine-tuning vs instruction tuning vs RLHF/DPO — conceptually and mathematically, not just as menu options
- **RL foundations behind RLHF**: policy gradients and reward modeling — enough to see that fine-tuning via human feedback is fundamentally a reinforcement learning problem (deep RL itself — robotics, game-playing — stays out of scope unless you want to go further later)
- **Parameter-efficient fine-tuning (LoRA, QLoRA)** — understand the math of low-rank adaptation, not just the library call
- Quantization: what int8/int4 actually do to a model and the tradeoffs involved
- Retrieval-augmented generation: embeddings, vector databases, chunking strategy, retrieval evaluation
- Prompt engineering as a discipline: few-shot, chain-of-thought, structured output
- **Evaluating generative systems**: hallucination detection, factuality checks, benchmark design — this matters enormously for clinical use, where a confidently wrong answer is dangerous in a way it isn't elsewhere
- Agents: tool use, planning, multi-step reasoning

**Deliverable**: build a RAG-based clinical Q&A or note-summarization tool, fine-tune a small open model with LoRA on a domain-specific task, and rigorously evaluate both — not just eyeball the outputs.

---

## Phase 6 — MLOps & Production Engineering

_Weeks 28–31_

This phase is the one most "LLM fine-tuner" tracks skip entirely — and it's a big part of what makes someone an engineer rather than a notebook hobbyist.

- Experiment tracking (Weights & Biases or MLflow)
- Model serving: batching, latency optimization, exporting to ONNX/TensorRT
- Distillation and compression for deployment on constrained hardware
- Data pipelines and dataset versioning (e.g. DVC)
- Monitoring models in production: drift detection, performance decay over time
- **Software engineering practices for ML**: testing ML code (data validation, model regression tests), reproducibility (seeding, environment pinning), CI/CD adapted for models rather than plain code
- Enough distributed training theory (data parallelism, gradient accumulation) to understand how large-scale training works, even without the hardware to do it yourself

**Deliverable**: take one earlier model and actually deploy it as a served API with basic monitoring — not just a script that runs once in a notebook.

---

## Phase 7 — Research Skills

_Weeks 32–34_

- How to read a paper efficiently: abstract → figures → method → results, in that order
- Reproducing a paper's core result at small scale
- Staying current: arXiv, and the venues that matter for your interests (NeurIPS/ICML for general ML, ACL for NLP, MICCAI for medical imaging)

**Deliverable**: pick one seminal paper relevant to clinical AI and reproduce its core experiment at small scale.

---

## Phase 8 — Capstone: AI + Medicine

_Weeks 35–39_

- The landscape: clinical NLP (notes, ICD coding), medical imaging (radiology, pathology), EHR/time-series modeling (e.g. ICU deterioration prediction), multimodal fusion (e.g. radiology report generation — combining imaging + text + tabular EHR data), genomics (graph neural networks are worth an optional look here if you go this direction)
- **Federated learning & differential privacy**: how models get trained across multiple hospitals' data without the data ever leaving each institution, since patient data legally can't be centralized the way a normal dataset can. Differential privacy adds a mathematical guarantee limiting what a trained model can leak about any single patient. One of the most distinctly medicine-specific topics in this whole curriculum.
- **Causal inference**: prediction and causal effect estimation are different problems — confounding, propensity score matching, causal graphs. "Did the treatment work" is usually the real clinical question, and it's a causal one, not a predictive one — likely the single most relevant addition here given your background
- Regulatory and deployment realities: FDA Software as a Medical Device (SaMD) pathway basics, HIPAA, clinical validation study design, mandatory bias/fairness audits for clinical deployment, plus documentation practices increasingly expected alongside them — model cards (a model's intended use, limitations, performance across subgroups) and datasheets for datasets (provenance, collection process, known biases)
- Build an end-to-end project combining several earlier phases — a model, an explainability layer, rigorous evaluation, and ideally a small deployment.

**Deliverable**: a portfolio-worthy project with a written report — the kind of artifact you'd actually show in an AI+medicine job interview or a grad school application.

---

## Note — second-pass review

A second review checked for real gaps beyond the first draft: GPU/systems basics, generative models beyond transformers, RL foundations behind RLHF, software engineering practices for ML, and causal inference. All five are now written directly into the phase sections above — not bolted on separately.

The one worth knowing about specifically: **causal inference**, added to Phase 8. Prediction and causal effect estimation are different problems, and "did the treatment work" — the question clinicians actually care about — is a causal question, not a predictive one. Given your background, this is probably the single most relevant addition in the whole document.

For the real time estimate, see the week-by-week schedule below — **39 weeks** is the number that matters, not any of the earlier rough ranges.

---

## Tools & libraries by phase

**Framework choice: PyTorch only, not TensorFlow.** It dominates research (including nearly all medical AI papers you'll read) and increasingly production. Learning both at once slows you down for little benefit — once you deeply understand PyTorch's mental model, picking up TensorFlow's syntax later takes days, not months.

| Library                                                         | Where it shows up                                                                                                                         |
| --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| NumPy                                                           | Phases 1–3, constantly — your tool for every "build it from scratch" exercise before PyTorch takes over                                   |
| Pandas                                                          | Phase 1 (data handling), Phase 8 (clinical/EHR tabular data is Pandas-heavy)                                                              |
| Matplotlib / seaborn                                            | Throughout — loss curves, decision boundaries, attention maps, SHAP plots, confusion matrices. No dedicated phase; picked up as needed    |
| scikit-learn                                                    | Phase 1 — reference implementation to check your from-scratch models against, plus preprocessing and evaluation metrics used throughout   |
| PyTorch                                                         | Phase 2 onward — the primary framework                                                                                                    |
| Hugging Face (`transformers`, `datasets`, `peft`, `tokenizers`) | Phase 3 onward — used _after_ you've built a transformer from scratch, so it stops being a black box                                      |
| `captum`, `shap`, `lime`                                        | Phase 4 (XAI)                                                                                                                             |
| `bitsandbytes`, `faiss` / `chromadb`                            | Phase 5 — quantization and RAG                                                                                                            |
| Weights & Biases or MLflow, `onnx` / `onnxruntime`, FastAPI     | Phase 6 (MLOps)                                                                                                                           |
| `pydicom`, `nibabel`                                            | Phase 8 — reading real clinical imaging formats (DICOM, NIfTI) rather than pre-cleaned datasets                                           |
| Docker                                                          | Phase 6 — containerizing the served model API; also useful earlier for reproducible training environments                                 |
| AWS/GCP (GPU instances, S3/Cloud Storage)                       | Phases 2, 3, 6 — GPU compute for training runs too big for a laptop, storage for datasets and model checkpoints, hosting the deployed API |

---

## Where your full-stack background helps

Already known: FastAPI, Flask, Express, React, Next, Vue, Nuxt (JS/TS, backend and frontend). This has real bearing on two things:

- **Phase 6 (MLOps)**: the model-serving week doesn't need to teach FastAPI/Flask from scratch — it goes straight to the ML-specific parts: loading models efficiently, batching requests, managing memory across concurrent requests, async inference. Real time saved.
- **Deliverables can be full products, not scripts.** Most ML portfolios end up backend-only — a script or a bare API endpoint. A real React/Next frontend on the Phase 5 RAG tool and the Phase 8 capstone turns them into usable, demoable clinical tools instead of notebooks. "Builds a real product _and_ understands the model internals" is a rarer combination than either skill alone — worth leaning into for interviews or grad applications.
- **Cloud and containers**: the AWS/GCP depth and Docker/infrastructure experience from Ethnoscyber carry over directly too — no need to learn cloud fundamentals from scratch, just apply what's already known to ML-specific needs: GPU instances for training runs too big for a laptop (starting Phase 2), cloud storage for datasets and model checkpoints, and containerizing the served API in Phase 6. Prefer raw EC2/Compute Engine + Docker over a fully-managed platform like SageMaker or Vertex AI for now — managed platforms hide the infrastructure decisions, which works against the "build it yourself first" approach this whole curriculum is built on. Worth knowing those platforms exist; not where the learning should happen. For the training-heavy weeks specifically (Phase 2's CNN, Phase 3's GPT), Google Colab, Lambda Labs, RunPod, or Vast.ai are worth knowing as cheaper GPU options than raw AWS pricing.

---

## Week-by-week schedule

Starting point: ~7 hours/week (1 hour/day). 39 weeks total (~9 months). Missed days just stretch a week rather than breaking the plan.

### Phase 1 — Math & ML Foundations (Weeks 1–4)

- **Week 1**:
  - **Day 1 (tomorrow)**: vectors, matrices, dot products, and the neuron/weighted-sum diagram from our first conversation — this is the official start
  - Rest of week: matrix multiplication as a full layer (batched neurons), more hands-on practice
- **Week 2**: Calculus for ML — derivatives, partial derivatives, chain rule, gradients; intro to gradient descent
- **Week 3**: Probability & statistics — distributions, Bayes' theorem, MLE, expectation/variance; classical ML concepts (linear/logistic regression)
- **Week 4**: _Deliverable_ — implement linear regression and logistic regression from scratch in NumPy; cross-validation, evaluation metrics (precision/recall/ROC-AUC)

### Phase 2 — Deep Learning Core (Weeks 5–12)

- **Week 5**: Forward pass, backprop derivation, computational graphs
- **Week 6**: Build a tiny autograd engine from scratch
- **Week 7**: Implement a multi-layer perceptron from scratch in NumPy
- **Week 8**: PyTorch fundamentals — tensors, autograd, `nn.Module`, training loops, `DataLoader`
- **Week 9**: Weight initialization, batch/layer norm, dropout, weight decay, LR scheduling
- **Week 10**: CNNs — convolution/pooling mechanics, ResNet family
- **Week 11**: RNNs/LSTMs/GRUs, vanishing gradients; GPU/systems basics (memory, mixed precision)
- **Week 12**: _Deliverable_ — train a CNN on a medical imaging dataset with experiment tracking; debugging deep nets

### Phase 3 — Transformers & Modern Architectures (Weeks 13–17)

- **Week 13**: Derive and implement scaled dot-product attention from scratch
- **Week 14**: Full transformer architecture — multi-head attention, positional encoding, layer norm placement, feedforward blocks
- **Week 15**: Build a small GPT from scratch, begin training on a toy corpus
- **Week 16**: Tokenization (BPE/WordPiece/SentencePiece), pretraining objectives (MLM vs CLM), encoder/decoder/encoder-decoder
- **Week 17**: Vision transformers (ViT); overview of VAEs, GANs, diffusion models; _deliverable_ — finish and train your from-scratch GPT

### Phase 4 — Explainable AI (Weeks 18–21)

- **Week 18**: Feature attribution — SHAP, LIME, integrated gradients
- **Week 19**: Visual explanation — Grad-CAM, saliency maps, attention visualization
- **Week 20**: Calibration and uncertainty quantification; counterfactual explanations; light look at adversarial robustness
- **Week 21**: Fairness/bias auditing; _deliverable_ — apply XAI to your Phase 2 and Phase 3 models, write an interpretation report

### Phase 5 — Applied Generative AI & NLP (Weeks 22–27)

- **Week 22**: Pretraining vs fine-tuning vs instruction tuning vs RLHF/DPO; RL foundations (policy gradients, reward modeling)
- **Week 23**: LoRA/QLoRA math; quantization (int8/int4)
- **Week 24**: RAG — embeddings, vector databases, chunking, retrieval evaluation
- **Week 25**: Prompt engineering as a discipline; agents — tool use, planning
- **Week 26**: Evaluating generative systems — hallucination detection, factuality, benchmark design
- **Week 27**: _Deliverable_ — build a RAG-based clinical Q&A/summarization tool, LoRA fine-tune a small model, evaluate both rigorously

### Phase 6 — MLOps & Production Engineering (Weeks 28–31)

- **Week 28**: Experiment tracking (W&B/MLflow); data pipelines and dataset versioning
- **Week 29**: Model serving — batching, latency optimization, ONNX/TensorRT export
- **Week 30**: Distillation/compression; software engineering practices for ML (testing, reproducibility, CI/CD)
- **Week 31**: Monitoring in production — drift detection, performance decay; _deliverable_ — deploy a served API with monitoring

### Phase 7 — Research Skills (Weeks 32–34)

- **Week 32**: How to read papers efficiently; survey clinical AI papers, pick one to reproduce
- **Weeks 33–34**: _Deliverable_ — reproduce the paper's core result at small scale

### Phase 8 — Capstone: AI + Medicine (Weeks 35–39)

- **Week 35**: Landscape survey — clinical NLP, medical imaging, EHR/time-series, multimodal, genomics, federated learning & differential privacy; choose project direction
- **Week 36**: Regulatory realities (FDA SaMD, HIPAA, clinical validation design, model cards/datasheets); causal inference — confounding, propensity matching, causal graphs
- **Weeks 37–38**: Build the end-to-end capstone project (model + XAI layer + evaluation + mini deployment)
- **Week 39**: Finalize and write up the portfolio report

Missed days don't break this — they just stretch a week. What matters is not skipping the order, since each phase leans on the one before it.

---

## Tracking progress on GitHub

You already know git, so this is just the structure that fits this specific plan — not a git tutorial.

**One repo for all 39 weeks**, organized by phase and week, since later deliverables lean on earlier ones (Phase 4's XAI work runs against the Phase 2 and Phase 3 models, for example — you'll want them in the same place).

```
ai-engineer-journey/
├── README.md
├── PROGRESS.md
├── phase-1-foundations/
│   ├── week-01-vectors-matrices/
│   ├── week-02-calculus/
│   ├── week-03-probability/
│   └── week-04-linear-logistic-regression/
├── phase-2-deep-learning-core/
│   ├── week-05-backprop/
│   ├── week-06-autograd-engine/
│   ├── week-07-mlp-from-scratch/
│   ├── week-08-pytorch-fundamentals/
│   ├── week-09-normalization-regularization/
│   ├── week-10-cnns/
│   ├── week-11-rnns-gpu-basics/
│   └── week-12-cnn-medical-imaging/
├── phase-3-transformers/
│   ├── week-13-attention-from-scratch/
│   ├── week-14-full-transformer/
│   ├── week-15-gpt-from-scratch/
│   ├── week-16-tokenization/
│   └── week-17-vit-generative-models/
├── phase-4-xai/
│   ├── week-18-feature-attribution/
│   ├── week-19-visual-explanation/
│   ├── week-20-calibration-uncertainty/
│   └── week-21-fairness-audit/
├── phase-5-applied-genai-nlp/
│   ├── week-22-rlhf-foundations/
│   ├── week-23-lora-quantization/
│   ├── week-24-rag/
│   ├── week-25-prompting-agents/
│   ├── week-26-evaluation/
│   └── week-27-clinical-rag-tool/
├── phase-6-mlops/
│   ├── week-28-experiment-tracking/
│   ├── week-29-model-serving/
│   ├── week-30-compression-testing/
│   └── week-31-monitoring-deployment/
├── phase-7-research/
│   ├── week-32-paper-reading/
│   └── weeks-33-34-paper-reproduction/
└── phase-8-capstone/
    ├── week-35-landscape-survey/
    ├── week-36-regulatory-causal-inference/
    ├── weeks-37-38-capstone-build/
    └── week-39-writeup/
```

**Commit at the end of every study session** — even a half-finished exercise or just notes. Daily commits build an honest, visible record across the 9 months, and the contribution graph itself becomes part of the portfolio story, separate from the code.

**Make the repo public.** A public repo showing consistent, structured progress — real from-scratch implementations, not tutorial follow-alongs — is genuinely strong evidence for interviews or grad applications. It shows the process, not just a finished project at the end.

**README.md**: link back to (or summarize) this curriculum document so anyone looking at the repo understands the plan, not just the code.

**PROGRESS.md**: a checklist mirroring the week-by-week schedule above, checked off as you go — it's a separate file (`PROGRESS.md`) so you can drop it straight into the repo root.

**Posting cadence**: weekly, not daily. Daily would compete directly with the 1 hour/day already going into the actual studying, and the early weeks (Week 1–4 math) don't have much worth writing about every single day. End of each week: one LinkedIn recap — what got built, what clicked, what was hard — matching the weekly folders and the PROGRESS.md checklist already in place. Lighter, more frequent posts on X/Twitter if wanted, since the cost there is much lower. Save the most effort for the 8 phase deliverables (autograd engine, CNN on medical images, GPT from scratch, XAI report, capstone) — those are the genuinely strong posts, not "studied today."
