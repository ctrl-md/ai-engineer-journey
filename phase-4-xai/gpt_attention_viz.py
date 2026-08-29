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
"""

import torch.nn as nn
from torch import arange, triu, ones, optim, no_grad, tensor
from torch.utils.data import Dataset, DataLoader
import torch
import sys

sys.path.insert(0, "/home/claude/gpt_final")
from corpus import corpus


def tokenizer(data):
    char = sorted(set(data))
    vocab_size = len(char)
    ch_id = {ch: id for id, ch in enumerate(char)}
    id_ch = {id: ch for id, ch in enumerate(char)}
    tokens = [ch_id[ch] for ch in data]
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


vocab_size, ch_id, id_ch, all_tokens = tokenizer(corpus)
all_tokens_t = torch.tensor(all_tokens)
seq_len = 64
windows = [
    all_tokens_t[i : i + seq_len + 1]
    for i in range(0, len(all_tokens_t) - seq_len - 1, 4)
]
windows = torch.stack(windows)
x = windows[:, :-1]
y = windows[:, 1:]
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
loader = DataLoader(PatientDataset(x_train, y_train), batch_size=16, shuffle=True)

for epoch in range(150):
    gpt.train()
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        pred, _ = gpt(batch_x)
        loss = loss_fn(pred.reshape(-1, vocab_size), batch_y.reshape(-1))
        loss.backward()
        optimizer.step()

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
