"""Provider-agnostic text generation and embeddings, local-first.

The default backend is a locally running **Ollama** server (no API key, no cloud),
reached over its native HTTP API with nothing but the standard library. OpenAI and
Gemini remain available as optional, opt-in fallbacks when a key is configured.

Routing is centralized here so the rest of the app can ask for "generate text with
the configured provider" without caring whether that means Ollama, OpenAI or Gemini.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_LLM_MODEL = "llama3.1"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"


def _post_json(url: str, payload: dict[str, object], *, timeout: float) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted local URL)
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def ollama_generate(
    *,
    base_url: str | None,
    model: str,
    prompt: str,
    temperature: float = 0.2,
    timeout: float = 180.0,
) -> str:
    """Generate text from a local Ollama model via its native ``/api/generate``."""
    base = (base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    try:
        data = _post_json(
            f"{base}/api/generate",
            {
                "model": model or DEFAULT_OLLAMA_LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=timeout,
        )
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {base}. Is it running? (ollama serve). Details: {exc}"
        ) from exc
    return str(data.get("response") or "").strip()


def ollama_embed(
    *,
    base_url: str | None,
    model: str,
    text: str,
    timeout: float = 120.0,
) -> list[float] | None:
    """Embed a single piece of text with a local Ollama embedding model."""
    base = (base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    try:
        data = _post_json(
            f"{base}/api/embeddings",
            {"model": model or DEFAULT_OLLAMA_EMBED_MODEL, "prompt": text},
            timeout=timeout,
        )
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {base} for embeddings. Is it running? Details: {exc}"
        ) from exc
    embedding = data.get("embedding")
    return [float(x) for x in embedding] if embedding else None


class EmptyCompletionError(RuntimeError):
    """An upstream returned 200 with no content.

    NabuGate's router already treats this as a failed target and moves down the
    chain, so seeing it here means every rung produced nothing. Presenting it as
    a successful empty answer is how an empty generation becomes a style
    exemplar for the next one.
    """


def openai_generate(
    *,
    api_key: str | None,
    base_url: str | None,
    model: str,
    prompt: str,
    temperature: float = 0.2,
) -> str:
    """Generate text via the OpenAI (or any OpenAI-compatible) chat API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key or "not-needed", base_url=base_url or None)
    resp = client.chat.completions.create(
        model=model or "gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def generate_text(
    *,
    provider: str | None,
    prompt: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    project_id: str | None = None,
    region: str = "us-central1",
    temperature: float = 0.2,
) -> str:
    """Generate text with the configured provider.

    ``base_url`` is the Ollama endpoint and is only used for the local provider;
    OpenAI/Gemini ignore it and use their own (cloud) endpoints.
    """
    provider = (provider or "ollama").lower()
    if provider == "nabugate":
        # The gateway is OpenAI-wire-compatible, so this is the same call with a
        # different base_url. `model` is an alias (nabu-fast / nabu-smart) or an
        # agent name; the gateway picks the provider and handles fallback.
        text = openai_generate(
            api_key=api_key,
            base_url=base_url,
            model=model or "nabu-fast",
            prompt=prompt,
            temperature=temperature,
        )
        if not (text or "").strip():
            raise EmptyCompletionError(f"{model or 'nabu-fast'} returned an empty completion")
        return text
    if provider in ("ollama", "local"):
        return ollama_generate(base_url=base_url, model=model or DEFAULT_OLLAMA_LLM_MODEL, prompt=prompt, temperature=temperature)
    if provider == "gemini":
        from .provider_http import gemini_generate_content

        return gemini_generate_content(
            api_key=api_key,
            model=model or "gemini-2.5-flash-lite",
            prompt=prompt,
            project_id=project_id,
            region=region,
        )
    # OpenAI (cloud) — base_url left to the client default.
    return openai_generate(api_key=api_key, base_url=None, model=model or "gpt-4o-mini", prompt=prompt, temperature=temperature)
