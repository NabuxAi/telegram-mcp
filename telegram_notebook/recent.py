"""Format the most recent archive items into compact rows for browsing.

Pure function over the dicts ``Repository.timeline_items`` returns (newest first),
so both the bot and the MCP layer can render it and it is easy to unit-test.
"""

from __future__ import annotations

from typing import Any


def recent_rows(items: list[dict[str, Any]], *, limit: int = 10, snippet_chars: int = 120) -> list[dict[str, str]]:
    """Compact rows: ``{source, date, snippet, url}`` for the newest ``limit`` items."""
    rows: list[dict[str, str]] = []
    for item in items[:limit]:
        source = (item.get("channel_title") or item.get("channel_url") or "Unknown source").strip()
        date = (item.get("message_date") or "")[:10]
        snippet = " ".join((item.get("text") or "").split())[:snippet_chars]
        rows.append({
            "source": source,
            "date": date,
            "snippet": snippet,
            "url": (item.get("message_url") or "").strip(),
        })
    return rows
