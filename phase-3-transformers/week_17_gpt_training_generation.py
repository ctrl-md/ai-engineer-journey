"""
Week 17 -- Vision transformers (ViT) conceptual overview, VAEs/GANs/
diffusion models overview, and the Phase 3 deliverable: finishing
and properly training the GPT.

ViT: split an image into fixed-size patches, flatten and linearly
project each one into a token, prepend a CLS token, add positional
embeddings, feed through a standard (usually encoder-only) transformer
-- exactly the token embedding pipeline already built for GPT, just
using nn.Linear instead of nn.Embedding since pixel patches are
continuous, not discrete IDs like characters.

VAEs/GANs/diffusion: covered conceptually, not built here -- an
encoder-decoder that encodes to a distribution instead of a point
(VAE), two networks trained adversarially (GAN), and iterative
denoising from pure noise (diffusion).

This file is the actual deliverable: trained on a richer, less
repetitive corpus than Week 15's toy text, and -- genuinely new --
autoregressive generate() actually producing new text, one character
at a time, feeding the model's own predictions back in as input.

Result: the model exactly memorized the corpus opening line (loss
dropped to 0.09 on only ~1700 characters of training text), then
drifted into character-level-plausible but not-quite-real English --
an honest result for a toy-scale model, not a broken one.

tokenizer and PatientDataset are identical to earlier weeks -- imported
rather than re-defined. Transformer and GPT ARE redefined here on
purpose: this week adds dropout, a genuine change from Week 14/15's
versions, not a duplicate.

prepare_training_windows and train_gpt are the reusable core of this
week's deliverable -- Week 21 (attention viz) and Week 27 (LoRA
deliverable) both import these instead of re-implementing the same
tokenize+window+train pattern a third and fourth time.
"""

import os
import sys

import torch.nn as nn
from torch import arange, triu, ones, optim, no_grad, tensor
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
from week_15_gpt_from_scratch import tokenizer
from week_08_pytorch_fundamentals import PatientDataset
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

    def forward(self, x, attn_mask):
        normed = self.norm1(x)
        output, _ = self.attention(normed, normed, normed, attn_mask=attn_mask)
        x = x + output
        x = x + self.dropout(self.feedforward(self.norm2(x)))
        return x


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

    def forward(self, tokens):
        seq_len = tokens.shape[-1]
        positions = arange(seq_len)
        x = self.token_embeddings(tokens) + self.positional_embeddings(positions)
        attn_mask = triu(ones(seq_len, seq_len), diagonal=1).bool()
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.final_norm(x)
        x = self.output_proj(x)
        return x


def window_tokens(all_tokens, seq_len, stride=4):
    """Build shift-by-one training windows from an already-tokenized
    sequence. Split out from prepare_training_windows so callers with
    their own tokenization (e.g. Week 27, reusing a saved vocabulary)
    can reuse just the windowing step."""
    windows = [
        all_tokens[i : i + seq_len + 1]
        for i in range(0, len(all_tokens) - seq_len - 1, stride)
    ]
    windows = torch.stack(windows)
    x = windows[:, :-1]
    y = windows[:, 1:]
    return x, y


def prepare_training_windows(text, seq_len, val_fraction=0.1):
    """Tokenize text and build shift-by-one windows for next-token
    training. Returns vocab_size, ch_id, id_ch, x_train, y_train."""
    vocab_size, ch_id, id_ch, all_tokens = tokenizer(text)
    all_tokens = torch.tensor(all_tokens)
    x, y = window_tokens(all_tokens, seq_len)
    n = len(x)
    x_train, y_train = (
        x[: int(n * (1 - val_fraction))],
        y[: int(n * (1 - val_fraction))],
    )
    return vocab_size, ch_id, id_ch, x_train, y_train


def train_gpt(
    gpt,
    x_train,
    y_train,
    vocab_size,
    epochs,
    batch_size=16,
    lr=0.001,
    print_every: int | None = 30,
):
    """Simple training loop -- no validation split or checkpointing
    (Week 15 already covers that pattern); handles GPT variants whose
    forward() returns a tuple (e.g. Week 21's need_weights version)
    as well as ones that return a plain tensor."""
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(gpt.parameters(), lr=lr)
    loader = DataLoader(
        PatientDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )

    for epoch in range(epochs):
        gpt.train()
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = gpt(batch_x)
            pred = output[0] if isinstance(output, tuple) else output
            loss = loss_fn(pred.reshape(-1, vocab_size), batch_y.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if print_every and ((epoch + 1) % print_every == 0 or epoch == 0):
            print(f"epoch {epoch+1}: avg loss = {total_loss/len(loader):.4f}")
    return gpt


def generate(model, tokens, num_new_tokens, max_seq_len):
    model.eval()
    for _ in range(num_new_tokens):
        context = tokens[-max_seq_len:]
        context = tensor([context])
        with no_grad():
            pred = model(context)
            pred = pred[0, -1].argmax(dim=-1)
            tokens.append(pred.item())
    return tokens


if __name__ == "__main__":
    vocab_size, ch_id, id_ch, x_train, y_train = prepare_training_windows(
        corpus, seq_len=64
    )
    print(f"corpus length: {len(corpus)} characters, vocab_size: {vocab_size}")
    print(f"training examples: {len(x_train)}")

    gpt = GPT(
        vocab_size=vocab_size,
        d_model=64,
        max_seq_len=64,
        num_heads=4,
        dropout_rate=0.1,
        d_ff=128,
        num_layers=3,
    )
    gpt = train_gpt(gpt, x_train, y_train, vocab_size, epochs=150)

    # now actually generate text from a seed prompt
    seed_text = "The human heart"
    seed_tokens = [ch_id[c] for c in seed_text]
    generated = generate(gpt, seed_tokens.copy(), num_new_tokens=150, max_seq_len=64)
    generated_text = "".join(id_ch[t] for t in generated)
    print(f"\n--- Generated text ---\n{generated_text}")
