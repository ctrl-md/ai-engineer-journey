"""
Week 21 -- Applying attention visualization to a real GPT prediction.
Part of the interpretation report deliverable (see
week-21-interpretation-report.md).

No extra computation needed -- attention weights are already produced
as part of the normal forward pass (need_weights=True on
nn.MultiheadAttention). Given "...complaining of fatigu", the model
correctly predicts 'e' (completing "fatigue"), and its attention
concentrates on the last few letters of the word it's actively
completing -- a sensible, locally-grounded pattern, not noise.

tokenizer, PatientDataset, prepare_training_windows, and train_gpt are
all identical to Week 15/8/17 -- imported rather than re-defined.
Transformer and GPT ARE redefined here on purpose: this week adds
need_weights support to extract real attention weights, a genuine
change from Week 17's version, not a duplicate. (train_gpt already
handles GPT variants whose forward() returns a tuple, like this one.)
"""

import os
import sys

import torch.nn as nn
from torch import arange, triu, ones, no_grad, tensor
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
from week_17_gpt_training_generation import prepare_training_windows, train_gpt
from corpus import corpus


class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout_rate, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_mask, need_weights=False):
        normed = self.norm1(x)
        output, weights = self.attention(
            normed,
            normed,
            normed,
            attn_mask=attn_mask,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        x = x + output
        x = x + self.dropout(self.feedforward(self.norm2(x)))
        return x, weights


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        max_seq_len,
        num_heads,
        dropout_rate,
        d_ff,
        num_layers,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.positional_embeddings = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                Transformer(d_model, num_heads, dropout_rate, d_ff)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, need_weights=False):
        seq_len = tokens.shape[-1]
        positions = arange(seq_len)
        x = self.token_embeddings(tokens) + self.positional_embeddings(positions)
        attn_mask = triu(ones(seq_len, seq_len), diagonal=1).bool()
        all_weights = []
        for block in self.blocks:
            x, weights = block(x, attn_mask, need_weights=need_weights)
            all_weights.append(weights)
        x = self.final_norm(x)
        x = self.output_proj(x)
        return x, all_weights


def visualize_attention(gpt, ch_id, id_ch):
    # pick a real sentence from the corpus and inspect attention for the LAST position's prediction
    gpt.eval()
    sample_text = "A patient walks into the clinic complaining of fatigu"
    sample_tokens = tensor([[ch_id[c] for c in sample_text]])

    with no_grad():
        output, all_weights = gpt(sample_tokens, need_weights=True)
        predicted_char = id_ch[output[0, -1].argmax(dim=-1).item()]

    print(f"input: '{sample_text}'")
    print(f"predicted next character: '{predicted_char}'")

    # layer 0's attention: which positions did the LAST token (predicting next char) attend to most?
    last_position_attention = all_weights[0][
        0, -1
    ]  # (seq_len,) -- attention from last query position over all keys
    top5 = torch.topk(last_position_attention, 5)
    print(
        f"\ntop 5 positions the model attended to (layer 1), for predicting '{predicted_char}':"
    )
    for idx, weight in zip(top5.indices.tolist(), top5.values.tolist()):
        print(f"  position {idx} ('{sample_text[idx]}'): weight={weight:.4f}")

    torch.save(
        {
            "sample_text": sample_text,
            "predicted_char": predicted_char,
            "top5_indices": top5.indices.tolist(),
            "top5_weights": top5.values.tolist(),
        },
        "gpt_attn_result.pt",
    )


if __name__ == "__main__":
    vocab_size, ch_id, id_ch, x_train, y_train = prepare_training_windows(
        corpus, seq_len=64
    )
    gpt = GPT(
        vocab_size=vocab_size,
        d_model=64,
        max_seq_len=64,
        num_heads=4,
        dropout_rate=0.1,
        d_ff=128,
        num_layers=3,
    )
    gpt = train_gpt(gpt, x_train, y_train, vocab_size, epochs=150, print_every=None)
    visualize_attention(gpt, ch_id, id_ch)
