"""Retrieval primitives for Phase 9: BM25, a real semantic embedding retriever,
reciprocal rank fusion, and the ranking metrics used to score them.

The semantic retriever is `fastembed` (ONNX Runtime, no torch) with
`minishlab/potion-retrieval-32M`, a static embedding model distilled from a larger
transformer (Model2Vec): embedding is a token lookup plus pooling, not a transformer
forward pass, so it's genuinely semantic (trained to put related text nearby in
vector space) without the per-query inference cost that comes with a full encoder.

Two things were tried first and ruled out:
- `sentence-transformers` (pulls in torch): installing it timed out after 3 minutes
  in this sandboxed environment with no prior download cache.
- `fastembed` with the default `BAAI/bge-small-en-v1.5` transformer model: it
  installs fine (ONNX Runtime alone, ~20MB), but measured at ~9 texts/second on this
  machine's 16 CPU cores -- fine for the ~1,300 synthetic chunks, but the real
  6-company sample alone chunks into ~60,000 paragraphs, which would take roughly
  two hours to embed. `potion-retrieval-32M` measured at ~1,900 texts/second on the
  same corpus (no transformer layers to run at inference time), bringing that down
  to well under a minute, with comparable semantic separation in a spot check
  (query-answer cosine 0.62 vs. query-unrelated-text cosine 0.03).

Calibration note: `fastembed==0.9.0` declares `huggingface-hub<2.0` but imports an
internal module (`_snapshot_download`) that huggingface-hub removed in its 1.x
release, so installing fastembed alone (which resolved huggingface-hub 1.33.0) broke
on import. Fixed by pinning `huggingface-hub==0.36.2` in requirements.txt.

Persisting the embeddings: `SemanticRetriever`'s `cache_path` loads/saves the
embedding matrix as a plain `.npz` (numpy's native archive) -- no new dependency, and
the right size for this corpus (~1,300 chunks): a real vector DB (FAISS, Chroma, ...)
would add an approximate index meant for millions of vectors, when this project
already does brute-force cosine search on purpose (see above). The cache stores a
sha256 of the embedded texts alongside the matrix, so a corpus change invalidates it
automatically rather than silently serving stale embeddings -- a one-shot script like
step12_evaluate_retrievers doesn't need this (it embeds once and exits), but a
long-lived process restarted often (src.chatbot_server) would otherwise re-embed
every restart for no reason.
"""

import hashlib
import math
import re
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi


TOKEN_RE = re.compile(r"[a-z0-9]+")
SEMANTIC_MODEL_NAME = "minishlab/potion-retrieval-32M"

_model: TextEmbedding | None = None


def _content_hash(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _load_cache(path: Path, chunk_ids: list[str], texts: list[str]) -> np.ndarray | None:
    """Return the cached doc_matrix if `path` exists and matches chunk_ids/texts
    exactly, else None (missing, corrupt, or stale -- all treated as a cache miss)."""
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        if data["content_hash"].item() != _content_hash(texts):
            return None
        if list(data["chunk_ids"]) != chunk_ids:
            return None
        return data["doc_matrix"]
    except Exception:
        return None


def _save_cache(path: Path, chunk_ids: list[str], texts: list[str], doc_matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")  # numpy appends .npz unless the name already ends with it
    np.savez(tmp, chunk_ids=np.array(chunk_ids), content_hash=np.array(_content_hash(texts)), doc_matrix=doc_matrix)
    tmp.replace(path)


def _get_model() -> TextEmbedding:
    """Load the embedding model once and reuse it across every retriever built."""
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=SEMANTIC_MODEL_NAME)
    return _model


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


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

    def top_score(self, query: str) -> float:
        """Return the best (unbounded) BM25 score for `query`, or 0.0 for an empty corpus."""
        scores = self._bm25.get_scores(tokenize(query))
        return float(max(scores)) if len(scores) else 0.0


class SemanticRetriever:
    """Cosine similarity over real sentence embeddings; see module docstring.

    Pass `cache_path` to persist the embedding matrix as a .npz and skip
    re-embedding on the next construction with the same chunk_ids/texts (see the
    module docstring's "Persisting the embeddings" note).
    """

    def __init__(self, chunk_ids: list[str], texts: list[str], cache_path: Path | None = None) -> None:
        self.chunk_ids = chunk_ids
        cached = _load_cache(cache_path, chunk_ids, texts) if cache_path is not None else None
        if cached is not None:
            self.doc_matrix = cached
            self.loaded_from_cache = True
            return
        embeddings = np.array(list(_get_model().embed(texts, batch_size=256)), dtype=np.float32)
        self.doc_matrix = _normalize(embeddings)
        self.loaded_from_cache = False
        if cache_path is not None:
            _save_cache(cache_path, chunk_ids, texts, self.doc_matrix)

    def rank(self, query: str) -> list[str]:
        """Return chunk_ids sorted best-first for `query`."""
        q_vec = np.array(next(iter(_get_model().embed([query]))), dtype=np.float32)
        q_vec = _normalize(q_vec[np.newaxis, :])[0]
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
