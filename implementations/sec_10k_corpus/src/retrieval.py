"""Retrieval primitives for Phase 9: BM25, a lightweight "dense" vector retriever,
reciprocal rank fusion, and the ranking metrics used to score them.

No neural embedding model backs the dense retriever: installing sentence-transformers
(which pulls in torch) was tried and timed out after 3 minutes in this sandboxed
environment (see PLAN.md Phase 9's "Calibration note"). TF-IDF + cosine similarity is
used instead -- a classic vector-space retriever, "dense" in the sense that query and
document each become one fixed-length numeric vector compared by a single score,
unlike BM25's sparse term-count scoring. This is a deliberate, documented
simplification, not a silent substitution, and it needs only numpy.
"""

import math
import re
from collections import Counter

import numpy as np
from rank_bm25 import BM25Okapi


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class Bm25Retriever:
    """Thin wrapper around rank_bm25.BM25Okapi, keyed by chunk_id."""

    def __init__(self, chunk_ids: list[str], texts: list[str]) -> None:
        self.chunk_ids = chunk_ids
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])

    def rank(self, query: str) -> list[str]:
        """Return chunk_ids sorted best-first for `query`."""
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)
        return [self.chunk_ids[i] for i in order]


class TfidfRetriever:
    """Pure-numpy TF-IDF vectors with cosine similarity; see module docstring."""

    def __init__(self, chunk_ids: list[str], texts: list[str], max_vocab: int = 8000) -> None:
        self.chunk_ids = chunk_ids
        token_lists = [tokenize(t) for t in texts]

        doc_freq: Counter = Counter()
        for tokens in token_lists:
            doc_freq.update(set(tokens))
        # Sort by (-count, term) rather than Counter.most_common(): most_common()
        # breaks count ties by insertion order, which depends on `set(tokens)`'s
        # hash-randomized iteration order above and made the vocab (and therefore
        # recall/MRR/nDCG) vary slightly between runs -- the ranking by count is
        # deterministic, this just makes the tie-break deterministic too.
        vocab = [term for term, _ in sorted(doc_freq.items(), key=lambda kv: (-kv[1], kv[0]))[:max_vocab]]
        self.vocab_index = {term: i for i, term in enumerate(vocab)}

        n_docs = len(texts)
        self.idf = np.array(
            [math.log((n_docs + 1) / (doc_freq[term] + 1)) + 1.0 for term in vocab],
            dtype=np.float32,
        )
        self.doc_matrix = np.stack([self._vectorize(tokens) for tokens in token_lists])

    def _vectorize(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocab_index), dtype=np.float32)
        for term, count in Counter(tokens).items():
            idx = self.vocab_index.get(term)
            if idx is not None:
                vec[idx] = (1.0 + math.log(count)) * self.idf[idx]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def rank(self, query: str) -> list[str]:
        """Return chunk_ids sorted best-first for `query`."""
        q_vec = self._vectorize(tokenize(query))
        scores = self.doc_matrix @ q_vec
        order = np.argsort(-scores)
        return [self.chunk_ids[i] for i in order]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Combine several best-first chunk_id rankings into one via RRF (standard k=60)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda c: -scores[c])


def recall_at_k(ranking: list[str], gold_ids: set[str], k: int) -> float:
    return 1.0 if set(ranking[:k]) & gold_ids else 0.0


def reciprocal_rank(ranking: list[str], gold_ids: set[str]) -> float:
    for i, chunk_id in enumerate(ranking):
        if chunk_id in gold_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranking: list[str], gold_ids: set[str], k: int) -> float:
    """Binary-relevance nDCG@k (one gold chunk_id set per question)."""
    dcg = sum(1.0 / math.log2(i + 2) for i, c in enumerate(ranking[:k]) if c in gold_ids)
    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
