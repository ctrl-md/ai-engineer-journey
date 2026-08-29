"""
Week 24 -- RAG: embeddings, vector databases, chunking, retrieval
evaluation.

embed: bag-of-words vectors (since no pretrained embedding model is
reachable in this environment) -- one dimension per vocabulary word,
counting occurrences. Simple, but genuinely meaningful for cosine
similarity: documents sharing vocabulary end up with similar vectors.

cosine_similarity: dot product normalized by both vectors' magnitudes
-- direction only, length ignored, so a long document chunk and a
short one aren't penalized just for differing in size.

retrieve: embeds a query, scores it against every document via cosine
similarity, returns the top k using torch.topk (values AND original
indices in one call).

evaluate_retrieval: Precision@k and Recall@k, direct reuse of Phase
1's precision/recall formulas -- "was this chunk actually relevant"
standing in for "was this prediction actually correct".
"""

import statistics
from torch import tensor, float32, topk


def embed(text, vocab):
    words = text.lower().split()
    vector = []
    for word in vocab:
        vector.append(words.count(word))
    return tensor(vector, dtype=float32)


def cosine_similarity(vec1, vec2):
    num = vec1 @ vec2
    denom = vec1.norm() * vec2.norm()
    return num / denom


def retrieve(query, documents, vocab, k):
    embed_query = embed(query, vocab)
    scores = []
    for document in documents:
        embed_document = embed(document, vocab)
        scores.append(cosine_similarity(embed_query, embed_document))
    scores = tensor(scores)
    top_scores = topk(scores, k)
    top_documents = []
    for index in top_scores.indices:
        top_documents.append(documents[index.item()])
    return top_documents


def evaluate_retrieval(test_queries, documents, vocab, k):
    precision_list = []
    recall_list = []

    for query, relevant_indices in test_queries:
        retrieved_documents = retrieve(query, documents, vocab, k)
        retrieved_indices = [documents.index(doc) for doc in retrieved_documents]
        relevant_retrieved_count = len(
            [i for i in retrieved_indices if i in relevant_indices]
        )

        precision = relevant_retrieved_count / len(retrieved_documents)
        recall = relevant_retrieved_count / len(relevant_indices)
        precision_list.append(precision)
        recall_list.append(recall)

    average_precision = statistics.mean(precision_list)
    average_recall = statistics.mean(recall_list)
    return average_precision, average_recall


if __name__ == "__main__":
    from knowledge_base import documents

    all_words = set()
    for doc in documents:
        all_words.update(doc.lower().split())
    vocab = sorted(all_words)
    print(f"vocab size: {len(vocab)}")

    query = "what medications treat high blood pressure"
    results = retrieve(query, documents, vocab, k=2)
    print(f"\nquery: '{query}'")
    print("top 2 retrieved documents:")
    for doc in results:
        print(f"  - {doc}")

    # ground truth: (query, indices of documents that are actually relevant)
    test_queries = [
        ("what medications treat high blood pressure", [0]),
        ("first line treatment for diabetes", [1]),
        ("symptoms of lung infection", [4]),
        ("why do kidney patients need different drug doses", [2]),
    ]
    avg_precision, avg_recall = evaluate_retrieval(test_queries, documents, vocab, k=2)
    print(f"\nAverage Precision@2: {avg_precision:.4f}")
    print(f"Average Recall@2: {avg_recall:.4f}")
