from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(".env")


def _int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def _str_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


@dataclass(slots=True)
class Settings:
    app_name: str
    data_dir: Path
    db_path: Path
    media_dir: Path
    telegram_api_id: int | None
    telegram_api_hash: str | None
    telegram_session_string: str | None
    telegram_session_name: str
    telegram_bot_token: str | None
    openai_api_key: str | None
    gemini_api_key: str | None
    nabugate_base_url: str
    nabugate_api_key: str | None
    vertex_project_id: str | None
    vertex_region: str | None
    vertex_index_id: str | None
    vertex_endpoint_id: str | None
    vertex_deployed_index_id: str | None
    telegram_proxy_host: str | None
    telegram_proxy_port: int | None
    telegram_proxy_type: str | None # 'http', 'socks5', 'mtproto'
    transcription_provider: str
    transcription_model: str
    embedding_provider: str
    embedding_model: str
    llm_provider: str
    llm_model: str
    ollama_base_url: str
    chunk_size: int
    chunk_overlap: int
    default_result_limit: int
    web_api_token: str | None
    miniapp_url: str | None
    podcast_tts_model: str
    podcast_llm_model: str | None

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)


def _default_provider() -> str:
    """Prefer the gateway whenever it is configured.

    A deployment that has never set NABUGATE_API_KEY keeps its old behaviour
    rather than failing on startup; one that sets it stops holding provider keys
    without also having to set three separate *_PROVIDER variables.
    """
    return "nabugate" if os.environ.get("NABUGATE_API_KEY", "").strip() else "ollama"


@lru_cache
def get_settings() -> Settings:
    load_dotenv(override=True)
    settings = Settings(
        app_name=_str_env("APP_NAME", "Telegram Notebook") or "Telegram Notebook",
        data_dir=Path(_str_env("DATA_DIR", "data") or "data"),
        db_path=Path(_str_env("DB_PATH", "data/store.db") or "data/store.db"),
        media_dir=Path(_str_env("MEDIA_DIR", "data/media") or "data/media"),
        telegram_api_id=_int_env("TELEGRAM_API_ID"),
        telegram_api_hash=_str_env("TELEGRAM_API_HASH"),
        telegram_session_string=_str_env("TELEGRAM_SESSION_STRING"),
        telegram_session_name=_str_env("TELEGRAM_SESSION_NAME", "telegram-notebook")
        or "telegram-notebook",
        telegram_bot_token=_str_env("TELEGRAM_BOT_TOKEN"),
        openai_api_key=_str_env("OPENAI_API_KEY"),
        gemini_api_key=_str_env("GEMINI_API_KEY"),
        # The gateway is the only AI credential this project should need. The
        # two keys above stay readable so an existing deployment keeps working,
        # but nothing new should be pointed at them.
        nabugate_base_url=_str_env("NABUGATE_BASE_URL", "https://gate.nabuxai.com/v1")
        or "https://gate.nabuxai.com/v1",
        nabugate_api_key=_str_env("NABUGATE_API_KEY"),
        vertex_project_id=_str_env("VERTEX_PROJECT_ID"),
        vertex_region=_str_env("VERTEX_REGION", "us-central1") or "us-central1",
        vertex_index_id=_str_env("VERTEX_INDEX_ID"),
        vertex_endpoint_id=_str_env("VERTEX_ENDPOINT_ID"),
        vertex_deployed_index_id=_str_env("VERTEX_DEPLOYED_INDEX_ID"),
        telegram_proxy_host=_str_env("TELEGRAM_PROXY_HOST"),
        telegram_proxy_port=_int_env("TELEGRAM_PROXY_PORT"),
        telegram_proxy_type=_str_env("TELEGRAM_PROXY_TYPE"),
        # Local-first defaults: transcription via local Whisper, embeddings + chat
        # via a local Ollama server. Cloud providers stay available when selected.
        transcription_provider=(
            _str_env("TRANSCRIPTION_PROVIDER", "local") or "local"
        ).lower(),
        transcription_model=_str_env(
            "TRANSCRIPTION_MODEL",
            "base",
        )
        or "base",
        embedding_provider=(_str_env("EMBEDDING_PROVIDER", _default_provider()) or _default_provider()).lower(),
        embedding_model=_str_env("EMBEDDING_MODEL", "nomic-embed-text")
        or "nomic-embed-text",
        llm_provider=(_str_env("LLM_PROVIDER", _default_provider()) or _default_provider()).lower(),
        llm_model=_str_env("LLM_MODEL", "llama3.1") or "llama3.1",
        ollama_base_url=_str_env("OLLAMA_BASE_URL", "http://localhost:11434")
        or "http://localhost:11434",
        chunk_size=_int_env("CHUNK_SIZE") or 900,
        chunk_overlap=_int_env("CHUNK_OVERLAP") or 120,
        default_result_limit=_int_env("DEFAULT_RESULT_LIMIT") or 8,
        web_api_token=_str_env("WEB_API_TOKEN"),
        # Public HTTPS URL where /miniapp is served (for the Telegram Mini App button).
        miniapp_url=_str_env("MINIAPP_URL"),
        # TTS engine for /podcast. "edge" needs no API key (default); "openai"/
        # "gemini"/"elevenlabs" use the matching provider key.
        podcast_tts_model=(_str_env("PODCAST_TTS_MODEL", "edge") or "edge").lower(),
        podcast_llm_model=_str_env("PODCAST_LLM_MODEL"),
    )
    settings.ensure_directories()
    return settings


def _env_safe(key: str, value: str | None) -> str:
    """Reject values that would inject extra assignments into ``.env``.

    Values reach here straight from ``POST /api/settings``, and the writer below
    emits ``KEY=value`` one per line — so a value containing a newline defines a
    second variable. Overwriting ``WEB_API_TOKEN`` that way bypasses auth on the
    next reload.
    """
    if value is None:
        return ""
    if "\n" in value or "\r" in value or "\x00" in value:
        raise ValueError(f"invalid value for {key}: line breaks are not allowed")
    return value


def upsert_env_values(updates: dict[str, str | None]) -> None:
    updates = {key: _env_safe(key, value) for key, value in updates.items()}
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    pending = dict(updates)
    output: list[str] = []

    for line in lines:
        if not line or line.lstrip().startswith("#") or "=" not in line:
            output.append(line)
            continue

        key, _, _ = line.partition("=")
        if key in pending:
            value = pending.pop(key)
            if value is None:
                output.append(f"{key}=")
            else:
                output.append(f"{key}={value}")
        else:
            output.append(line)

    for key, value in pending.items():
        output.append(f"{key}={'' if value is None else value}")

    ENV_PATH.write_text("\n".join(output) + "\n", encoding="utf-8")
    get_settings.cache_clear()


def provider_credentials(settings: Settings, provider: str | None) -> tuple[str | None, str | None]:
    """The (api_key, base_url) a provider needs, chosen in exactly one place.

    Four call sites used to repeat the same ternary, so adding the gateway would
    otherwise have meant adding a fifth branch to each of them.
    """
    provider = (provider or "").lower()
    if provider == "nabugate":
        return settings.nabugate_api_key, settings.nabugate_base_url
    if provider == "gemini":
        return settings.gemini_api_key, None
    if provider in ("ollama", "local"):
        return None, settings.ollama_base_url
    return settings.openai_api_key, None


#: What to ask the gateway for when no explicit model is configured.
#: ``notebook-embed`` is this project's own single-rung alias — see the comment
#: on NABUGATE_EMBED_DIM for why it must not be ``nabu-embed``.
NABUGATE_DEFAULT_LLM_MODEL = "nabu-fast"
NABUGATE_DEFAULT_EMBED_MODEL = "notebook-embed"
#: The gateway's transcription alias. Multilingual and timestamped — the same
#: chain ReelMind uses — so a Persian voice note comes back as Persian text
#: rather than a transliteration.
NABUGATE_DEFAULT_TRANSCRIBE_MODEL = "nabu-transcribe"


def model_for(settings: Settings, provider: str | None, role: str) -> str:
    """The model name for a role, defaulting to the gateway's aliases."""
    provider = (provider or "").lower()
    configured = {
        "embedding": settings.embedding_model,
        "transcription": settings.transcription_model,
    }.get(role, settings.llm_model)
    if provider != "nabugate":
        return configured
    # An operator who set EMBEDDING_MODEL for Ollama, or TRANSCRIPTION_MODEL for
    # Gemini, must not have that value silently sent to the gateway as an alias
    # it has never heard of.
    default = {
        "embedding": NABUGATE_DEFAULT_EMBED_MODEL,
        "transcription": NABUGATE_DEFAULT_TRANSCRIBE_MODEL,
    }.get(role, NABUGATE_DEFAULT_LLM_MODEL)
    return configured if configured.startswith(("nabu-", "notebook-")) else default
