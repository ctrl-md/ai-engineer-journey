"""
Week 15 -- Build a small GPT from scratch, train on a toy corpus.
Phase 3's deliverable: causal (masked) self-attention so the model
can't peek at future tokens, a full GPT architecture (token +
positional embeddings, stacked causal transformer blocks, output
projection), character-level tokenization, and a real training run.
"""

import torch.nn as nn
from torch import arange, triu, ones, optim, no_grad
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from statistics import mean
import torch


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
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class Transformer(nn.Module):
    """Causal transformer block -- same as Week 14, plus attn_mask support."""
    def __init__(self, d_model, num_heads, dropout_rate, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_mask):
        normed = self.norm1(x)
        output, _ = self.attention(normed, normed, normed, attn_mask=attn_mask)
        x = x + output
        x = x + self.dropout(self.feedforward(self.norm2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, num_heads, dropout_rate, d_ff, num_layers):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.positional_embeddings = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Transformer(d_model, num_heads, dropout_rate, d_ff) for _ in range(num_layers)])
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


def data_split(x, y):
    size = len(x)
    train_index = int(0.70 * size)
    val_index = int(0.85 * size)
    return x[:train_index], y[:train_index], x[train_index:val_index], y[train_index:val_index], x[val_index:], y[val_index:]


def train(x_train, y_train, x_val, y_val, batch_size, vocab_size, d_model, max_seq_len,
          num_heads, dropout_rate, d_ff, num_layers, learning_rate, weight_decay, step_size, gamma, epochs):
    dataset = PatientDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size, shuffle=True)
    gpt = GPT(vocab_size, d_model, max_seq_len, num_heads, dropout_rate, d_ff, num_layers)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(gpt.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    performance = []
    best_state = None
    best_acc = 0

    for epoch in range(epochs):
        gpt.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            y_pred = gpt(batch_x)
            loss_fn = loss(y_pred.reshape(-1, vocab_size), batch_y.reshape(-1))
            loss_fn.backward()
            optimizer.step()

        with no_grad():
            gpt.eval()
            pred_val = gpt(x_val)
            val_loss_fn = loss(pred_val.reshape(-1, vocab_size), y_val.reshape(-1))
            val_acc = pred_val.argmax(dim=-1) == y_val
            val_acc = val_acc.float().mean().item() * 100

        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = deepcopy(gpt.state_dict())

        performance.append({"train_loss": loss_fn.item(), "val_loss": val_loss_fn.item(),
                             "val_acc": val_acc, "epoch": epoch + 1})

    gpt.load_state_dict(best_state)
    print(f"Best validation accuracy: {best_acc:.4f}")
    return gpt, performance


def evaluate(x_test, y_test, batch_size, gpt, vocab_size):
    dataset = PatientDataset(x_test, y_test)
    loader = DataLoader(dataset, batch_size, shuffle=True)
    loss = nn.CrossEntropyLoss()
    loss_list = []
    accuracy_list = []
    performance = []
    gpt.eval()

    for batch_num, (batch_x, batch_y) in enumerate(loader):
        test_pred = gpt(batch_x)
        loss_fn = loss(test_pred.reshape(-1, vocab_size), batch_y.reshape(-1))
        test_acc = test_pred.argmax(dim=-1) == batch_y
        test_acc = test_acc.float().mean().item() * 100
        loss_list.append(loss_fn.item())
        accuracy_list.append(test_acc)
        performance.append({"test_loss": loss_fn.item(), "test_accuracy": test_acc, "batch": batch_num + 1})

    print(f"Average test loss: {mean(loss_list):.4f}")
    print(f"Average test accuracy: {mean(accuracy_list):.4f}")
    return performance


if __name__ == "__main__":
    text = "the cat sat on the mat. the dog sat on the log. " * 20
    vocab_size, ch_id, id_ch, all_tokens = tokenizer(text)
    all_tokens = torch.tensor(all_tokens)

    seq_len = 16
    windows = [all_tokens[i:i + seq_len + 1] for i in range(0, len(all_tokens) - seq_len - 1, 4)]
    windows = torch.stack(windows)
    x = windows[:, :-1]
    y = windows[:, 1:]

    x_train, y_train, x_val, y_val, x_test, y_test = data_split(x, y)

    gpt, performance = train(x_train, y_train, x_val, y_val, batch_size=8, vocab_size=vocab_size,
                              d_model=32, max_seq_len=seq_len, num_heads=4, dropout_rate=0.1, d_ff=64, num_layers=2,
                              learning_rate=0.001, weight_decay=0, step_size=30, gamma=0.5, epochs=60)

    test_performance = evaluate(x_test, y_test, batch_size=8, gpt=gpt, vocab_size=vocab_size)