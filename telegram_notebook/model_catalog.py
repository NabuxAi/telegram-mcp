from __future__ import annotations

from typing import Any

from .provider_http import list_gemini_models, list_openai_models


class ModelCatalogService:
    def list_models(
        self,
        *,
        provider: str,
        api_key: str | None,
        capability: str | None = None,
    ) -> list[dict[str, Any]]:
        if not api_key:
            raise RuntimeError(f"API key for provider '{provider}' is missing")

        if provider == "gemini":
            return list_gemini_models(api_key=api_key, capability=capability)

        return list_openai_models(api_key=api_key, capability=capability)
