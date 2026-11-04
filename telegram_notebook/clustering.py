"""Offline topic clustering over stored chunk embeddings.

Groups archived chunks into topics using a single pass of greedy
cosine-similarity clustering against running centroids — pure Python, no
third-party dependencies, and deterministic for a given input order. Cluster
labels are derived from the most frequent significant words in each cluster,
so labelling works without an LLM call.

Because chunks already carry embeddings from ingestion, clustering needs no
network access.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from typing import Any

from .embeddings import cosine_similarity

# Small multilingual stop-word set so cluster labels surface meaningful terms.
_STOPWORDS = {
    # English
    "the", "and", "for", "are", "was", "were", "this", "that", "with", "from",
    "have", "has", "had", "not", "but", "you", "your", "our", "their", "his",
    "her", "its", "they", "them", "out", "about", "into", "over", "than", "then",
    "what", "when", "which", "who", "will", "would", "can", "could", "should",
    # Persian (common function words)
    "این", "آن", "که", "را", "با", "برای", "از", "به", "در", "است", "هست",
    "شد", "شده", "می", "هم", "یا", "تا", "بر", "هر", "اما", "اگر", "روی", "های",
}

_TOKEN_RE = re.compile(r"[A-Za-z؀-ۿ][A-Za-z؀-ۿ]+")


def top_terms(texts: list[str], *, k: int = 4) -> list[str]:
    """Most frequent significant terms across ``texts`` (for labelling)."""
    counter: Counter[str] = Counter()
    for text in texts:
        for token in _TOKEN_RE.findall((text or "").lower()):
            if len(token) >= 3 and token not in _STOPWORDS:
                counter[token] += 1
    return [term for term, _ in counter.most_common(k)]


def cluster_embeddings(
    items: list[dict[str, Any]],
    *,
    threshold: float = 0.6,
    max_clusters: int = 20,
) -> list[list[int]]:
    """Greedy single-pass clustering. Returns lists of item indices, one per cluster.

    Each item must have an ``embedding`` (list of floats). An item joins the most
    similar existing cluster if cosine similarity to its centroid is >= ``threshold``;
    otherwise it starts a new cluster (until ``max_clusters``, after which it is
    attached to the nearest cluster).
    """
    centroids: list[list[float]] = []
    sums: list[list[float]] = []
    members: list[list[int]] = []

    for idx, item in enumerate(items):
        emb = item.get("embedding")
        if not emb:
            continue

        best_i, best_sim = -1, -1.0
        for ci, centroid in enumerate(centroids):
            sim = cosine_similarity(emb, centroid)
            if sim > best_sim:
                best_sim, best_i = sim, ci

        # An archive re-embedded with a different model holds vectors of two
        # sizes. cosine_similarity returns 0.0 for those, but the max_clusters
        # branch below would still attach one to a centroid it cannot be summed
        # with — which used to abort the whole run on zip(strict=True).
        if best_i >= 0 and len(emb) != len(centroids[best_i]):
            best_i, best_sim = -1, -1.0

        if best_i >= 0 and (best_sim >= threshold or len(centroids) >= max_clusters):
            members[best_i].append(idx)
            sums[best_i] = [s + e for s, e in zip(sums[best_i], emb, strict=True)]
            n = len(members[best_i])
            centroids[best_i] = [s / n for s in sums[best_i]]
        else:
            centroids.append(list(emb))
            sums.append(list(emb))
            members.append([idx])

    return members


def build_label_prompt(texts: list[str], *, max_items: int = 6, max_chars: int = 300) -> str:
    """Prompt asking the model for a short topic label over a cluster's snippets."""
    samples = "\n".join(f"- {t.strip()[:max_chars]}" for t in texts[:max_items] if t.strip())
    return (
        "Below are related snippets from a personal archive. Give a single short topic "
        "label (2 to 5 words, no quotes or trailing punctuation) in the snippets' main "
        "language. Reply with ONLY the label.\n\n"
        f"Snippets:\n{samples}\n\nLabel:"
    )


def parse_topic_label(response: str | None) -> str:
    """Clean an LLM label reply: first non-empty line, no surrounding quotes, capped."""
    for line in (response or "").splitlines():
        cleaned = line.strip().strip("\"'").strip()
        if cleaned:
            return cleaned[:60]
    return ""


def label_cluster(texts: list[str], *, generate: Callable[[str], str]) -> str:
    """Produce an LLM topic label for a cluster using the injected ``generate`` callable."""
    return parse_topic_label(generate(build_label_prompt(texts)))


def build_topics(
    items: list[dict[str, Any]],
    *,
    threshold: float = 0.6,
    max_clusters: int = 20,
    min_size: int = 1,
    namer: Callable[[list[str]], str] | None = None,
) -> list[dict[str, Any]]:
    """Cluster ``items`` and return topic summaries sorted by size (desc).

    Each topic: ``{label, size, sample, sources}``. ``items`` carry ``embedding``,
    ``text`` and optionally ``channel_title``/``channel_url``. When ``namer`` is given
    it labels each cluster from its texts (e.g. via an LLM); on empty result or error
    it falls back to the keyword-frequency label.
    """
    clusters = cluster_embeddings(items, threshold=threshold, max_clusters=max_clusters)
    topics: list[dict[str, Any]] = []
    for member_indices in clusters:
        if len(member_indices) < min_size:
            continue
        texts = [items[i].get("text") or "" for i in member_indices]
        label = ""
        if namer is not None:
            try:
                label = (namer(texts) or "").strip()
            except Exception:
                label = ""
        if not label:
            terms = top_terms(texts)
            label = "، ".join(terms) if terms else "Untitled topic"
        sources = sorted({
            (items[i].get("channel_title") or items[i].get("channel_url"))
            for i in member_indices
            if items[i].get("channel_title") or items[i].get("channel_url")
        })
        sample = next((t.strip() for t in texts if t.strip()), "")
        topics.append({
            "label": label,
            "size": len(member_indices),
            "sample": sample[:200],
            "sources": sources,
        })
    topics.sort(key=lambda t: t["size"], reverse=True)
    return topics
