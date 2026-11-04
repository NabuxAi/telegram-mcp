"""Chronological view of the archive — the time-based counterpart to topic clustering.

Groups dated content items into calendar buckets (by day or month) and returns
period summaries. Pure functions over plain dicts with no third-party
dependencies, so the whole thing is deterministic and trivially testable.

Message dates are stored as ISO 8601 strings, so bucketing is just a prefix of
the date: ``YYYY-MM`` for months, ``YYYY-MM-DD`` for days.
"""

from __future__ import annotations

from typing import Any

_GRANULARITIES = {"day": 10, "month": 7}  # prefix length of an ISO date string


def period_key(iso_date: str, granularity: str) -> str | None:
    """Bucket key for an ISO 8601 date, or None if it doesn't start with a date."""
    if not iso_date:
        return None
    width = _GRANULARITIES.get(granularity, 7)
    key = iso_date[:width]
    # A valid bucket must start with YYYY-MM (digits + dashes), e.g. "2026-06".
    if len(key) < 7 or key[4] != "-" or not key[:4].isdigit() or not key[5:7].isdigit():
        return None
    return key


def build_timeline(
    items: list[dict[str, Any]],
    *,
    granularity: str = "month",
) -> list[dict[str, Any]]:
    """Group ``items`` by calendar period, newest first.

    Each item carries ``message_date`` (ISO string) and optionally ``text`` and
    ``channel_title``/``channel_url``. Returns one dict per period:
    ``{period, count, sources, sample}``.
    """
    buckets: dict[str, dict[str, Any]] = {}
    for item in items:
        key = period_key(item.get("message_date") or "", granularity)
        if key is None:
            continue
        bucket = buckets.get(key)
        if bucket is None:
            bucket = buckets[key] = {"period": key, "count": 0, "_sources": set(), "sample": ""}
        bucket["count"] += 1
        source = item.get("channel_title") or item.get("channel_url")
        if source:
            bucket["_sources"].add(source)
        if not bucket["sample"]:
            text = (item.get("text") or "").strip()
            if text:
                bucket["sample"] = text[:160]

    periods = []
    for key in sorted(buckets, reverse=True):
        bucket = buckets[key]
        periods.append({
            "period": bucket["period"],
            "count": bucket["count"],
            "sources": sorted(bucket.pop("_sources")),
            "sample": bucket["sample"],
        })
    return periods
