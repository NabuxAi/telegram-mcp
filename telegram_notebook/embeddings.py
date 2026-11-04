from __future__ import annotations

import math
from collections.abc import Sequence

from .llm import ollama_embed
from .provider_http import gemini_embed_text

#: The width of the ``notebook-embed`` alias on NabuGate.
#:
#: gemini-embedding-001 behind that alias defaults to 3072, so the width is only
#: 1536 because we ask for it on every request. Everything already written to
#: ``chunks.embedding`` is 1536, and a vector of another width in that column
#: would not fail — it would return a plausible cosine that means nothing.
NABUGATE_EMBED_DIM = 1536


class EmbeddingDimensionError(RuntimeError):
    """An embedding came back a different width than the stored index uses."""


class EmbeddingService:
    def __init__(self, *, provider: str, api_key: str | None, model: str, base_url: str | None = None) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.client = None

    @property
    def enabled(self) -> bool:
        # Local providers (Ollama) need no API key; cloud providers do.
        if self.provider in ("ollama", "local"):
            return True
        return self.api_key is not None

    def _get_client(self):
        if not self.api_key:
            return None
        if self.client is None:
            if self.provider != "gemini":
                from openai import OpenAI

                # NabuGate speaks the OpenAI wire protocol, so it is the same
                # client with a different base_url and the project token.
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
        return self.client

    def embed(self, text: str, *, task_type: str | None = None, project_id: str | None = None, region: str = "us-central1") -> list[float] | None:
        if not self.api_key and not project_id:
            # If no API key, we might be using gcloud auth which needs project_id
            pass

        if self.provider in ("ollama", "local"):
            return ollama_embed(base_url=self.base_url, model=self.model, text=text)

        if self.provider == "gemini":
            return gemini_embed_text(
                api_key=self.api_key,
                model=self.model,
                text=text,
                task_type=task_type,
                project_id=project_id,
                region=region
            )

        client = self._get_client()
        if not client:
            return None

        kwargs: dict[str, object] = {
            "model": self.model,
            "input": text,
            "encoding_format": "float",
        }
        if self.provider == "nabugate":
            # Required, not optional: the alias is single-rung precisely so that
            # nothing can quietly write a second geometry into the column, and
            # the model behind it returns 3072 unless asked otherwise.
            kwargs["dimensions"] = NABUGATE_EMBED_DIM

        response = client.embeddings.create(**kwargs)
        vector = list(response.data[0].embedding)

        if self.provider == "nabugate" and len(vector) != NABUGATE_EMBED_DIM:
            # Refuse rather than store. A failed embed is retryable; a column
            # holding two embedding spaces is not detectable after the fact.
            raise EmbeddingDimensionError(
                f"{self.model} returned {len(vector)} dimensions, expected {NABUGATE_EMBED_DIM}"
            )

        return vector


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
