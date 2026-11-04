"""Render archive statistics (from ``Repository.archive_stats``) to a readable summary.

Pure function so it is easy to unit-test; the bot and MCP layers just pass the
dict through and send the resulting text.
"""

from __future__ import annotations

from typing import Any


def format_stats(stats: dict[str, Any]) -> str:
    """Plain-text overview of an archive's counts and date range."""
    if not stats.get("items"):
        return "Your archive is empty. Ingest a channel or forward messages to get started."

    lines = [
        f"Items: {stats['items']}",
        f"Sources: {stats['sources']}",
        f"Tags: {stats['tags']}",
    ]

    by_kind = stats.get("by_kind") or {}
    if by_kind:
        kinds = ", ".join(f"{kind}: {count}" for kind, count in by_kind.items())
        lines.append(f"By type: {kinds}")

    first, last = stats.get("first_date"), stats.get("last_date")
    if first and last:
        lines.append(f"Date range: {first[:10]} → {last[:10]}")

    return "\n".join(lines)
