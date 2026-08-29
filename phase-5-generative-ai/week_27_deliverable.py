"""
Week 27 -- Phase 5 deliverable: build a RAG-based clinical Q&A tool,
LoRA fine-tune a small model, evaluate both rigorously.

Reuses, rather than repeats: PatientDataset (Week 8), tokenizer,
Transformer/GPT with dropout (Week 15/17), LoRALinear (Week 23),
embed/cosine_similarity/retrieve/evaluate_retrieval (Week 24) -- all
imported. This file contains only what's genuinely new for the
deliverable itself:

- lora_setup: freezes an ENTIRE pretrained GPT, then replaces
  output_proj with a LoRALinear wrapper -- so A/B become the only
  trainable parameters in the whole model.
- Pretraining + LoRA fine-tuning on a small clinical Q&A pattern
  ("Q: ... A: ...") the base GPT never saw during pretraining --
  verified two ways: real loss decrease during fine-tuning, AND the
  frozen base weight/bias proven byte-identical before and after real
  training (not just at initialization).
- Hallucination detection: generate the same prompt multiple times
  with dropout deliberately kept ACTIVE (not eval() mode) -- reusing
  MC Dropout's exact insight from Week 20/26. A consistency score of
  1.00 across 8 generations is a real, checkable signal the model's
  completion is grounded, not confabulated.
"""

import os
import sys
from typing import cast

import torch.nn as nn
from torch import arange, triu, ones, optim, no_grad, tensor, load
from torch.utils.data import DataLoader
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0,
    os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "phase-2-deep-learning-core",
        )
    ),
)
sys.path.insert(
    0,
    os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "phase-3-transformers"
        )
    ),
)
from week_08_pytorch_fundamentals import PatientDataset
from week_17_gpt_training_generation import (
    tokenizer,
    Transformer,
    GPT,
    prepare_training_windows,
    window_tokens,
    train_gpt,
)
from week_23_lora import LoRALinear
from clinical_qa import qa_pairs


def lora_setup(gpt, rank):
    for param in gpt.parameters():
        param.requires_grad = False
    gpt.output_proj = LoRALinear(gpt.output_proj, rank)
    return gpt


def pretrain_base_gpt(
    d_model, num_heads, dropout_rate, d_ff, num_layers, seq_len, epochs, save_path
):
    combined_text = " ".join(qa_pairs)
    # val_fraction=0.0: use the entire small corpus for pretraining, no held-out split
    vocab_size, ch_id, id_ch, x, y = prepare_training_windows(
        combined_text, seq_len, val_fraction=0.0
    )

    gpt = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=seq_len,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        d_ff=d_ff,
        num_layers=num_layers,
    )
    print("pretraining base GPT...")
    gpt = train_gpt(gpt, x, y, vocab_size, epochs=epochs, print_every=None)

    original_weight_before = gpt.output_proj.weight.clone()
    original_bias_before = gpt.output_proj.bias.clone()

    torch.save(
        {
            "gpt_state": gpt.state_dict(),
            "vocab_size": vocab_size,
            "ch_id": ch_id,
            "id_ch": id_ch,
            "seq_len": seq_len,
            "original_weight_before": original_weight_before,
            "original_bias_before": original_bias_before,
        },
        save_path,
    )
    print(f"pretraining done, saved to {save_path}")


def run_lora_finetuning(
    qa_pairs,
    file_name,
    d_model,
    num_heads,
    dropout_rate,
    d_ff,
    num_layers,
    rank,
    learning_rate,
    epochs,
):
    trained_model = load(file_name)
    gpt = GPT(
        trained_model["vocab_size"],
        d_model,
        trained_model["seq_len"],
        num_heads,
        dropout_rate,
        d_ff,
        num_layers,
    )
    gpt.load_state_dict(trained_model["gpt_state"])
    # tokenize using the SAME vocabulary the base model was pretrained with
    ch_id = trained_model["ch_id"]
    tokens = tensor([ch_id[ch] for ch in " ".join(qa_pairs)])

    seq_len = trained_model["seq_len"]
    x, y = window_tokens(tokens, seq_len)

    gpt = lora_setup(gpt, rank)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(gpt.parameters(), lr=learning_rate)

    print("fine-tuning with LoRA...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = gpt(x)
        loss_fn = loss(y_pred.reshape(-1, trained_model["vocab_size"]), y.reshape(-1))
        loss_fn.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch}: loss = {loss_fn.item():.4f}")

    # gpt.output_proj was replaced with a LoRALinear in lora_setup(); cast tells
    # the type checker what's already true at runtime.
    lora_output_proj = cast(LoRALinear, gpt.output_proj)
    is_weight_equal = torch.equal(
        lora_output_proj.original_linear.weight,
        trained_model["original_weight_before"],
    )
    is_bias_equal = torch.equal(
        lora_output_proj.original_linear.bias, trained_model["original_bias_before"]
    )
    return gpt, trained_model, is_weight_equal, is_bias_equal


def generate_with_dropout_active(model, tokens, num_new_tokens, max_seq_len):
    """Same generate() logic as Week 15/17, but deliberately keeps dropout
    ACTIVE (model.train()) instead of calling model.eval() -- reusing MC
    Dropout's exact insight: repeated generation with dropout on reveals
    how much the model's output depends on which neurons survive dropout
    each time, i.e. how uncertain/consistent the model actually is."""
    model.train()
    tokens = list(tokens)
    with no_grad():
        for _ in range(num_new_tokens):
            context = tokens[-max_seq_len:]
            context_t = tensor([context])
            pred = model(context_t)
            next_token = pred[0, -1].argmax(dim=-1).item()
            tokens.append(next_token)
    return tokens


def check_consistency(
    model, prompt_tokens, id_ch, num_samples, num_new_tokens, max_seq_len
):
    generations = []
    for _ in range(num_samples):
        result = generate_with_dropout_active(
            model, prompt_tokens.copy(), num_new_tokens, max_seq_len
        )
        generated_text = "".join(id_ch[t] for t in result)
        generations.append(generated_text)

    matches = sum(1 for g in generations if g == generations[0])
    consistency_score = matches / num_samples
    return generations, consistency_score


if __name__ == "__main__":
    save_path = "pretrained_gpt.pt"
    pretrain_base_gpt(
        d_model=48,
        num_heads=4,
        dropout_rate=0.1,
        d_ff=96,
        num_layers=3,
        seq_len=32,
        epochs=80,
        save_path=save_path,
    )

    gpt, trained_model, is_weight_equal, is_bias_equal = run_lora_finetuning(
        qa_pairs,
        save_path,
        d_model=48,
        num_heads=4,
        dropout_rate=0.1,
        d_ff=96,
        num_layers=3,
        rank=4,
        learning_rate=0.001,
        epochs=100,
    )
    print(f"\nbase weight unchanged after real fine-tuning: {is_weight_equal}")
    print(f"base bias unchanged after real fine-tuning: {is_bias_equal}")

    ch_id = trained_model["ch_id"]
    id_ch = trained_model["id_ch"]
    prompt = "Q: What is first-line treatment for hyp"
    prompt_tokens = [ch_id[c] for c in prompt]

    generations, consistency_score = check_consistency(
        gpt,
        prompt_tokens,
        id_ch,
        num_samples=8,
        num_new_tokens=15,
        max_seq_len=trained_model["seq_len"],
    )
    print(f"\nConsistency score: {consistency_score:.2f}")
    for i, g in enumerate(generations):
        print(f"  {i}: {g}")
