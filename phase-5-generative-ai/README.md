# Phase 5 — Applied Generative AI & NLP

Weeks 22–27 of the curriculum. Full plan: [`../ai_engineer_curriculum.md`](../ai_engineer_curriculum.md)

**Phase complete.**

## What's covered

- **Pretraining, fine-tuning, instruction tuning, RLHF/DPO, RL foundations** (Week 22): pretraining reframed as something already done (the Phase 3 GPT); why RLHF is fundamentally different from every prior training loop (one whole-response reward, no per-token signal); reward modeling and policy gradients as the mechanism that turns that single reward into a weight update; DPO as a genuinely simpler, supervised-learning-shaped alternative. Purely conceptual — no code, per the curriculum's own framing.
- **LoRA/QLoRA, quantization** (Week 23): `LoRALinear` built from scratch — freezing an original layer entirely, training only a small low-rank correction — verified two ways (fresh output matches the original exactly, and only A/B show up as trainable parameters). Real parameter-count math: rank-8 LoRA on a 1000×1000 matrix trains 1.6% of a full fine-tune. Quantization covered conceptually.
- **RAG** (Week 24): bag-of-words embeddings, cosine similarity, `retrieve()` using `torch.topk`, and `evaluate_retrieval()` reusing Phase 1's Precision/Recall formulas directly.
- **Prompt engineering, agents** (Week 25): few-shot, chain-of-thought (a genuine computational advantage from causal attention, not just style), structured output, tool use (RAG as one specific instance of it), planning (chain-of-thought and tool calls, chained and repeated). Conceptual.
- **Evaluating generative systems** (Week 26): hallucination detection via consistency checking (direct reuse of MC Dropout's insight) and retrieval-grounding; benchmark contamination as Phase 1's test-set integrity problem at LLM scale. Conceptual.
- **The deliverable** (Week 27): a full pipeline — pretrain a small GPT, LoRA fine-tune it on a clinical Q&A pattern it never saw during pretraining (frozen base proven byte-identical before/after real training, not just at init), and a hallucination consistency check (dropout kept deliberately active, 8/8 identical generations). Built and debugged end to end, including a genuinely hard multi-round debugging arc on the fine-tuning pipeline itself.

## Files

- `week_23_lora.py` — `LoRALinear`, verified against the original layer and real parameter-count math
- `week_24_rag_retrieval.py` — `embed`, `cosine_similarity`, `retrieve`, `evaluate_retrieval`
- `week_27_deliverable.py` — the full pipeline: pretrain, LoRA fine-tune, hallucination consistency check. Imports `PatientDataset` (Week 8), `tokenizer`/`Transformer`/`GPT` (Week 15/17), `LoRALinear` (Week 23) rather than repeating them.
- `knowledge_base.py`, `clinical_qa.py` — data files (not week-numbered, same convention as `corpus.py` in Phase 3)

Weeks 22, 25, and 26 have no code files — purely conceptual, same convention as earlier purely-conceptual sections.

## Status

**Phase 5 complete.** Currently on: Phase 6 — MLOps & Production Engineering.
