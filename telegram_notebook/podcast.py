"""Generate an Audio Overview (podcast) from archive items via Podcastfy.

This adds NotebookLM's signature "Audio Overview" feature: turn a scope of the
archive (whole archive, a source, a tag, or a collection) into a short
conversational podcast.

Podcastfy is a heavy, optional dependency, so it is **not** imported at module
load — only lazily inside :func:`synthesize_podcast`. That keeps the bot (and
the test suite) importable without it; install the feature with
``pip install ".[podcast]"``.

:func:`build_podcast_source` is a pure function over the same item dicts that
``Repository.summary_items`` returns, so it is trivial to unit-test.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class PodcastUnavailable(RuntimeError):
    """Raised when the optional Podcastfy dependency is not installed."""


def build_podcast_source(
    scope_label: str,
    items: list[dict[str, Any]],
    *,
    max_items: int = 40,
    per_item_chars: int = 1500,
) -> str:
    """Flatten archive ``items`` into a single source document for the podcast.

    Items are capped (``max_items``) and truncated (``per_item_chars``) to keep
    the prompt — and the resulting audio — to a sane length and cost.
    """
    parts = [f"Telegram archive — {scope_label}", ""]
    for i, item in enumerate(items[:max_items], 1):
        source = (item.get("channel_title") or item.get("channel_url") or "Unknown source").strip()
        text = (item.get("text") or "").strip()[:per_item_chars]
        if not text:
            continue
        parts.append(f"[{i}] {source}: {text}")
    return "\n\n".join(parts).strip()


def synthesize_podcast(
    *,
    text: str,
    gemini_api_key: str | None = None,
    openai_api_key: str | None = None,
    tts_model: str = "edge",
    llm_model_name: str | None = None,
) -> str:
    """Generate an audio overview from ``text`` and return the path to the audio file.

    Provider keys are exported to the environment because Podcastfy (and the
    underlying litellm/TTS clients) read them from there. The LLM that writes the
    conversation transcript uses the Gemini key when available, otherwise OpenAI.

    Raises :class:`PodcastUnavailable` when Podcastfy is not installed, or
    ``RuntimeError`` when generation yields no audio.
    """
    try:
        from podcastfy.client import generate_podcast
    except Exception as exc:  # pragma: no cover - only without the optional extra
        raise PodcastUnavailable(
            'Audio Overview needs the optional dependency. Install it with: '
            'pip install "telegram-notebooklm-mvp[podcast]"'
        ) from exc

    if gemini_api_key:
        os.environ["GEMINI_API_KEY"] = gemini_api_key
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    # Pick the LLM that drafts the conversation. Podcastfy defaults to a Gemini
    # model, so we only need to override when falling back to OpenAI.
    if gemini_api_key:
        api_key_label: str | None = "GEMINI_API_KEY"
    elif openai_api_key:
        api_key_label = "OPENAI_API_KEY"
        llm_model_name = llm_model_name or "gpt-4o-mini"
    else:
        api_key_label = None

    audio_path = generate_podcast(
        text=text,
        tts_model=tts_model,
        llm_model_name=llm_model_name,
        api_key_label=api_key_label,
    )
    if not audio_path:
        raise RuntimeError("Podcast generation returned no audio file.")
    return audio_path
