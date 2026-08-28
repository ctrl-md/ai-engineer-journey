"""
Week 16 -- Tokenization, pretraining objectives, architecture choices.

A real, working Byte-Pair Encoding (BPE) implementation -- the exact
merge process worked through by hand on the "low/own/west" example
(counting adjacent pairs, merging the most frequent one, repeating).

The rest of this week was conceptual, reasoned through rather than
coded:
- MLM (BERT-style: predict a masked word using context from BOTH
  directions) vs CLM (GPT-style: predict the next token using only
  what came before -- what this whole GPT was built on)
- Encoder-only (no causal mask, full bidirectional attention --
  understanding/classification tasks like flagging a clinical note
  as urgent) vs decoder-only (causal, generates text) vs
  encoder-decoder (both -- e.g. translation)
"""

from collections import Counter


def get_pair_counts(corpus):
    """Count every adjacent pair of tokens across the corpus."""
    counts = Counter()
    for word, freq in corpus.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            counts[(symbols[i], symbols[i + 1])] += freq
    return counts


def merge_pair(pair, corpus):
    """Merge every occurrence of `pair` into one new token, across the corpus."""
    new_corpus = {}
    bigram = " ".join(pair)
    merged = "".join(pair)
    for word, freq in corpus.items():
        new_word = word.replace(bigram, merged)
        new_corpus[new_word] = freq
    return new_corpus


def train_bpe(corpus, num_merges):
    """Run BPE for a fixed number of merges, printing each step."""
    merges = []
    for step in range(num_merges):
        pair_counts = get_pair_counts(corpus)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=pair_counts.get)
        merges.append(best_pair)
        corpus = merge_pair(best_pair, corpus)
        print(
            f"merge {step + 1}: {best_pair} (count={pair_counts[best_pair]}) -> {corpus}"
        )
    return corpus, merges


if __name__ == "__main__":
    # same example worked through by hand: low x5, own x3, west x2
    # each word starts as individual characters, space-separated
    corpus = {
        "l o w": 5,
        "o w n": 3,
        "w e s t": 2,
    }
    final_corpus, merges = train_bpe(corpus, num_merges=3)
    print(f"\nmerges learned, in order: {merges}")
