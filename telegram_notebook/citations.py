"""NotebookLM-style inline citations for grounded ``/ask`` answers.

The answer model is told to cite each claim inline as ``[n]`` referencing the
numbered search results it was given (``Source [1]``, ``Source [2]``…). These
pure helpers parse those markers back out and render a numbered sources block
that lines up with the inline citations, so an answer reads like:

    Al Mouj دارای townhouse است [1] و قیمت‌ها اعلام شده‌اند [2].

    Sources:
    [1] Channel A — https://t.me/a/12
    [2] Channel B — https://t.me/b/7

Everything here is UI-agnostic and works on either ``SearchResult`` objects or
plain dicts, so it is trivially unit-testable.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

# Matches [1], [12], and grouped forms like [1, 2] or [۱،۲] (Western/Arabic commas).
_CITE_RE = re.compile(r"\[\s*(\d+(?:\s*[,،]\s*\d+)*)\s*\]")


def _get(result: Any, attr: str) -> Any:
    if isinstance(result, dict):
        return result.get(attr)
    return getattr(result, attr, None)


def cited_indices(answer: str, max_n: int) -> list[int]:
    """Return the distinct 1-based source numbers cited in ``answer``.

    Only numbers within ``1..max_n`` are kept (the model is asked not to invent
    citations, but we defend against it anyway). Result is sorted ascending.
    """
    found: list[int] = []
    for match in _CITE_RE.finditer(answer or ""):
        for part in re.split(r"[,،]", match.group(1)):
            part = part.strip()
            if part.isdigit():
                n = int(part)
                if 1 <= n <= max_n and n not in found:
                    found.append(n)
    return sorted(found)


def source_label(result: Any) -> str:
    """Human label for a search result (channel title, URL, or a fallback)."""
    return _get(result, "channel_title") or _get(result, "channel_url") or "Unknown source"


def format_sources_block(
    results: list[Any],
    indices: list[int] | None = None,
    *,
    limit: int | None = None,
    escape: Callable[[str], str] | None = None,
) -> str:
    """Render a numbered ``[n] label — url`` block.

    When ``indices`` is given (the citations actually used), only those sources
    are listed, preserving the answer's numbering. Otherwise every result is
    listed (capped by ``limit``) as a graceful fallback for answers that did not
    cite. ``escape`` is applied to the dynamic label/url (e.g. ``html.escape``).
    """
    esc = escape or (lambda s: s)
    if indices:
        chosen = [(i, results[i - 1]) for i in indices if 1 <= i <= len(results)]
    else:
        chosen = list(enumerate(results, 1))
        if limit is not None:
            chosen = chosen[:limit]
    lines = []
    for i, result in chosen:
        label = esc(str(source_label(result)))
        url = _get(result, "message_url")
        line = f"[{i}] {label}"
        if url:
            line += f" — {esc(str(url))}"
        lines.append(line)
    return "\n".join(lines)
