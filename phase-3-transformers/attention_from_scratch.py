"""
Week 13 -- Scaled dot-product attention, from scratch.
softmax and attention(Q, K, V), verified against hand-worked
calculations for a small "sat" attending to "The cat sat down" example.
"""

import torch

def softmax(scores):
    exp_scores = torch.exp(scores)
    sum_scores = exp_scores.sum(dim=-1, keepdim=True)
    return exp_scores/sum_scores

def attention(Q, K, V):
    num = Q @ K.T
    denom = Q.shape[-1] ** 0.5
    scores = num/denom
    return softmax(scores) @ V


if __name__ == "__main__":
    # "The cat sat down" -- sat's Query, and each word's Key/Value
    Q = torch.tensor([[1.0, 2.0]])          # sat's query
    K = torch.tensor([[0.0, 0.0],           # The
                       [2.0, 1.0],           # cat
                       [1.0, 1.0],           # sat
                       [0.5, 0.5]])          # down
    V = torch.tensor([[1.0, 0.0],           # The
                       [4.0, 2.0],           # cat
                       [0.0, 1.0],           # sat
                       [2.0, 2.0]])          # down

    raw_scores = Q @ K.T
    print(f"raw scores: {raw_scores}")  # matches hand calc [0, 4, 3, 1.5]

    output = attention(Q, K, V)
    print(f"attention output (scaled): {output}")