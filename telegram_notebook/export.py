"""Render an archive scope (whole archive, a source, or a tag) to a Markdown document.

Kept as a pure function over the same item dicts that ``Repository.summary_items``
returns, so it is trivial to unit-test and the bot layer only has to write the
string to a temp file and upload it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def build_markdown_export(scope_label: str, items: list[dict[str, Any]], *, generated_at: str | None = None) -> str:
    """Return a Markdown document for ``items`` under ``scope_label``."""
    stamp = generated_at or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Telegram Archive — {scope_label}",
        "",
        f"_Exported {stamp} · {len(items)} item(s)_",
        "",
    ]
    for i, item in enumerate(items, 1):
        source = (item.get("channel_title") or item.get("channel_url") or "Unknown source").strip()
        lines.append(f"## {i}. {source}")
        url = (item.get("message_url") or "").strip()
        if url:
            lines.append(f"[{url}]({url})")
        text = (item.get("text") or "").strip()
        if text:
            lines.append("")
            lines.append(text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
