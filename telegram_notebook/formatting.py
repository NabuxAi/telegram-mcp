"""Render LLM/Markdown output into something Telegram can display safely.

LLM answers (``/ask``, ``/summarize``, ``/digest``) come back as Markdown
(``**bold**``, ``## headings``, bullet lists, code fences). Sending that raw
inside a ``parse_mode=HTML`` message both fails to render the formatting and can
break delivery entirely when the text contains ``<``/``&``. ``telegramify-markdown``
converts Markdown to a valid Telegram **MarkdownV2** string and handles all the
escaping for us.

The import is guarded and conversion is wrapped so a missing dependency or a
malformed LLM payload can never crash message delivery — we fall back to plain
text (no ``parse_mode``) in that case.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised implicitly by import
    from telegramify_markdown import markdownify as _markdownify
except Exception:  # pragma: no cover - only when the dependency is missing
    _markdownify = None


def to_telegram_markdown(text: str) -> tuple[str, str | None]:
    """Convert Markdown ``text`` to Telegram MarkdownV2.

    Returns ``(rendered_text, parse_mode)``. ``parse_mode`` is ``"MarkdownV2"``
    on success, or ``None`` (send as plain text) when the library is unavailable
    or conversion fails.
    """
    if _markdownify is None:
        return text, None
    try:
        return _markdownify(text), "MarkdownV2"
    except Exception:
        logger.warning("Markdown→Telegram conversion failed; sending plain text", exc_info=True)
        return text, None
