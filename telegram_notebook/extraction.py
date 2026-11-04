"""Text extraction (OCR / PDF) for forwarded image and document media.

A thin wrapper over the Gemini multimodal extractor, mirroring
``TranscriptionService`` so the bot can construct a per-user instance and the
ingest orchestration can treat transcription and extraction uniformly.
"""

from __future__ import annotations

from pathlib import Path

from .provider_http import gemini_extract_document


class ExtractionService:
    def __init__(self, *, provider: str, api_key: str | None, model: str) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key

    @property
    def enabled(self) -> bool:
        # Extraction uses Gemini multimodal; only available with a Gemini key.
        return self.provider == "gemini" and self.api_key is not None

    def extract(self, source_path: Path) -> str:
        if not self.enabled:
            raise RuntimeError("A Gemini API key is required for document/image extraction")
        return gemini_extract_document(api_key=self.api_key, model=self.model, file_path=source_path)
