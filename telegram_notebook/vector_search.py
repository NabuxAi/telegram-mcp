"""Local semantic search: rank stored chunk embeddings against a query vector.

Pure functions over plain lists/dicts (only the standard library), so ranking is
deterministic and unit-testable without a database or network. This is the
in-process fallback used whenever no external vector index (Vertex AI) is
configured, so the archive is semantically searchable out of the box.

For large archives this is an O(n) scan; it is the natural precursor to a
pgvector/Qdrant backend, which would replace ``rank_by_cosine`` with an indexed
nearest-neighbour query while keeping the same call site.
"""

from __future__ import annotations

import json
import math
from typing import Any


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors; 0.0 for empty/mismatched."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):  # lengths verified equal above
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _as_vector(value: Any) -> list[float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    if isinstance(value, list) and value:
        return value
    return None


def rank_by_cosine(
    query_vector: list[float],
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
    embedding_key: str = "embedding_json",
) -> list[tuple[dict[str, Any], float]]:
    """Return the ``top_k`` candidates most similar to ``query_vector``.

    Each candidate is a dict carrying its embedding under ``embedding_key`` (a JSON
    string or a list of floats). Candidates without a usable embedding are skipped.
    Results are ``(candidate, score)`` pairs sorted by descending similarity.
    """
    if not query_vector or top_k <= 0:
        return []
    scored: list[tuple[dict[str, Any], float]] = []
    for candidate in candidates:
        vector = _as_vector(candidate.get(embedding_key))
        if vector is None:
            continue
        scored.append((candidate, cosine_similarity(query_vector, vector)))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
