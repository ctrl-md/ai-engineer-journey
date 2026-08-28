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
"""

import torch.nn as nn
from torch import arange, triu, ones, optim, no_grad, tensor
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import torch
from corpus import corpus


def tokenizer(data):
    char = sorted(set(data))
    vocab_size = len(char)
    ch_id = {ch: id for id, ch in enumerate(char)}
    id_ch = {id: ch for id, ch in enumerate(char)}
    tokens = []
    for ch in data:
        tokens.append(ch_id[ch])
    return vocab_size, ch_id, id_ch, tokens


class PatientDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


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


vocab_size, ch_id, id_ch, all_tokens = tokenizer(corpus)
all_tokens = torch.tensor(all_tokens)
print(f"corpus length: {len(corpus)} characters, vocab_size: {vocab_size}")

seq_len = 64
windows = [
    all_tokens[i : i + seq_len + 1] for i in range(0, len(all_tokens) - seq_len - 1, 4)
]
windows = torch.stack(windows)
x = windows[:, :-1]
y = windows[:, 1:]
print(f"training examples: {len(x)}")

n = len(x)
x_train, y_train = x[: int(n * 0.9)], y[: int(n * 0.9)]

gpt = GPT(
    vocab_size=vocab_size,
    d_model=64,
    max_seq_len=seq_len,
    num_heads=4,
    dropout_rate=0.1,
    d_ff=128,
    num_layers=3,
)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(gpt.parameters(), lr=0.001)
dataset = PatientDataset(x_train, y_train)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

epochs = 150
for epoch in range(epochs):
    gpt.train()
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        pred = gpt(batch_x)
        loss = loss_fn(pred.reshape(-1, vocab_size), batch_y.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 30 == 0 or epoch == 0:
        print(f"epoch {epoch+1}: avg loss = {total_loss/len(loader):.4f}")

# now actually generate text from a seed prompt
seed_text = "The human heart"
seed_tokens = [ch_id[c] for c in seed_text]
generated = generate(gpt, seed_tokens.copy(), num_new_tokens=150, max_seq_len=seq_len)
generated_text = "".join(id_ch[t] for t in generated)
print(f"\n--- Generated text ---\n{generated_text}")
