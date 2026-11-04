from __future__ import annotations

import asyncio
import html
import logging
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from . import citations, toolset
from .bot_api import TelegramBotApi
from .clustering import build_topics, label_cluster
from .config import Settings, get_settings, model_for, provider_credentials
from .db import Repository, connect
from .deleted_watcher import DeletedWatcherManager
from .embeddings import EmbeddingService
from .export import build_markdown_export
from .extraction import ExtractionService
from .formatting import to_telegram_markdown
from .jobs import JobWorker
from .llm import generate_text
from .logging_config import setup_logging
from .media import route_media
from .office import detect_office_kind, extract_office_text
from .pipeline import IngestionPipeline, build_media_text_extractor
from .podcast import PodcastUnavailable, build_podcast_source, synthesize_podcast
from .provider_http import gemini_generate_content
from .recent import recent_rows
from .rules import classify_ai_tags, match_tags
from .search import SearchService
from .stats import format_stats
from .telegram_backup import (
    count_messages,
    make_zip_file_resolver,
    parse_export,
    read_export,
    render_markdown,
)
from .telegram_client import (
    build_client_from_session_string,
    request_login_code,
    sign_in_with_code,
    sign_in_with_password,
)
from .timeline import build_timeline
from .transcription import TranscriptionService

logger = logging.getLogger(__name__)


# --- LLM routing (local-first via Ollama; cloud optional) --------------------
# These take ``settings`` explicitly so they work with any settings-like object.
# NOTE: the getattr default below is "gemini" only so legacy/test settings objects
# that predate llm_provider keep their cloud behavior. The real Settings object
# always carries llm_provider (default "ollama").
def _llm_kwargs_for(settings, user: dict | None) -> dict:
    """Generation kwargs for SearchService.generate_answer / summarize."""
    provider = (getattr(settings, "llm_provider", None) or "gemini").lower()
    api_key = None
    if provider == "gemini":
        api_key = (user.get("gemini_api_key") if user else None) or getattr(settings, "gemini_api_key", None)
    elif provider == "openai":
        api_key = (user.get("openai_api_key") if user else None) or getattr(settings, "openai_api_key", None)
    return {
        "provider": provider,
        "model": getattr(settings, "llm_model", None),
        "base_url": getattr(settings, "ollama_base_url", None),
        "api_key": api_key,
        "project_id": getattr(settings, "vertex_project_id", None),
        "region": getattr(settings, "vertex_region", None) or "us-central1",
    }


def _llm_callable_for(settings, user: dict | None):
    """A ``generate(prompt) -> str`` for tagging/clustering, or None if unavailable."""
    provider = (getattr(settings, "llm_provider", None) or "gemini").lower()
    if provider == "gemini":
        api_key = (user.get("gemini_api_key") if user else None) or getattr(settings, "gemini_api_key", None)
        if not api_key:
            return None
        # Reference the module global so tests can monkeypatch gemini_generate_content.
        return lambda prompt: gemini_generate_content(
            api_key=api_key,
            prompt=prompt,
            project_id=getattr(settings, "vertex_project_id", None),
            region=getattr(settings, "vertex_region", None) or "us-central1",
        )
    if provider == "nabugate":
        api_key, base_url = provider_credentials(settings, "nabugate")
        if not api_key:
            return None
        return lambda prompt: generate_text(
            provider="nabugate",
            model=model_for(settings, "nabugate", "llm"),
            prompt=prompt,
            api_key=api_key,
            base_url=base_url,
        )
    if provider == "openai":
        api_key = (user.get("openai_api_key") if user else None) or getattr(settings, "openai_api_key", None)
        if not api_key:
            return None
        return lambda prompt: generate_text(
            provider="openai", model=getattr(settings, "llm_model", None), prompt=prompt, api_key=api_key
        )
    # ollama / local — no API key required.
    return lambda prompt: generate_text(
        provider=provider,
        model=getattr(settings, "llm_model", None),
        prompt=prompt,
        base_url=getattr(settings, "ollama_base_url", None),
    )


def _llm_available_for(settings, user: dict | None) -> bool:
    return _llm_callable_for(settings, user) is not None


@dataclass(slots=True)
class BotServices:
    api: TelegramBotApi
    repository: Repository
    search_service: SearchService
    pipeline: IngestionPipeline
    settings: Settings


def build_services() -> BotServices:
    settings = get_settings()
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required to run the bot")

    repository = Repository(connect(settings.db_path))
    repository.init()

    embed_key, embed_base = provider_credentials(settings, settings.embedding_provider)
    embeddings = EmbeddingService(
        provider=settings.embedding_provider,
        api_key=embed_key,
        base_url=embed_base,
        model=model_for(settings, settings.embedding_provider, "embedding"),
    )
    tr_key, _ = provider_credentials(settings, settings.transcription_provider)
    _, tr_base = provider_credentials(settings, settings.transcription_provider)
    transcription = TranscriptionService(
        provider=settings.transcription_provider,
        api_key=tr_key,
        model=model_for(settings, settings.transcription_provider, "transcription"),
        base_url=tr_base,
    )
    pipeline = IngestionPipeline(
        settings=settings,
        repository=repository,
        transcription=transcription,
        embeddings=embeddings,
    )
    search_service = SearchService(repository, embeddings)
    api = TelegramBotApi(settings.telegram_bot_token)
    return BotServices(
        api=api,
        repository=repository,
        search_service=search_service,
        pipeline=pipeline,
        settings=settings,
    )


def normalize_phone(raw: str) -> str | None:
    table = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
    text = raw.translate(table)
    compact = "".join(c for c in text if c.isdigit())
    if not (10 <= len(compact) <= 15):
        return None
    return f"+{compact}"


def normalize_code(raw: str) -> str | None:
    if not raw:
        return None
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    english_digits = "0123456789"

    text = raw
    for i in range(10):
        text = text.replace(persian_digits[i], english_digits[i])
        text = text.replace(arabic_digits[i], english_digits[i])

    compact = "".join(c for c in text if c.isdigit())

    if 3 <= len(compact) <= 12:
        return compact
    return None


class NotebookBot:
    def __init__(self, services: BotServices) -> None:
        self.services = services
        self.offset: int | None = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.worker = JobWorker(self.services.repository, self._run_import_job)
        self.watcher = DeletedWatcherManager(
            repository=self.services.repository,
            build_client=self._build_watcher_client,
            notify=self._notify_recovered,
        )

    def run_forever(self) -> None:
        logger.info("Bot polling started")
        self.worker.start()
        miniapp_url = getattr(self.services.settings, "miniapp_url", None)
        if miniapp_url:
            try:
                self.services.api.set_chat_menu_button(url=miniapp_url, text="Open App")
            except Exception:
                logger.exception("Failed to set Mini App menu button")
        try:
            self.watcher.start_all()
        except Exception:
            logger.exception("Failed to start deleted-message watchers")
        while True:
            try:
                updates = self.services.api.get_updates(offset=self.offset, timeout=30)
                for update in updates:
                    self.offset = int(update["update_id"]) + 1
                    try:
                        self.handle_update(update)
                    except Exception:
                        logger.exception("Failed to handle update %s", update.get("update_id"))
            except KeyboardInterrupt:
                logger.info("Bot stopped by KeyboardInterrupt")
                break
            except Exception:
                logger.exception("Bot polling error")
                time.sleep(3)

    def _async_to_sync(self, coro, timeout: int = 60):
        try:
            return self.loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
        except TimeoutError as exc:
            raise RuntimeError("Operation timed out. Please check your connection or try again.") from exc

    def _api_key_for_user(self, user: dict | None, provider: str) -> str | None:
        if provider == "nabugate":
            # Deliberately not per-user: the gateway token is the project's, and
            # spend is attributed to the project rather than to whoever typed.
            return self.services.settings.nabugate_api_key
        if provider == "gemini":
            return (user.get("gemini_api_key") if user else None) or self.services.settings.gemini_api_key
        if provider == "openai":
            return self.services.settings.openai_api_key
        return None

    def _embedding_service_for_user(self, user: dict | None) -> EmbeddingService:
        provider = self.services.settings.embedding_provider
        model = (
            user.get("preferred_embedding_model")
            if user and user.get("preferred_embedding_model")
            else self.services.settings.embedding_model
        )
        _, base = provider_credentials(self.services.settings, provider)
        # A per-user embedding model is not a preference here, it is a hazard:
        # the stored index has one width, and a second model writes a second
        # geometry into the same column where it reads as a plausible cosine.
        # model_for pins the gateway's own single-rung alias.
        if provider == "nabugate":
            model = model_for(self.services.settings, provider, "embedding")
        return EmbeddingService(
            provider=provider,
            api_key=self._api_key_for_user(user, provider),
            model=model,
            base_url=base,
        )

    def _transcription_service_for_user(self, user: dict | None) -> TranscriptionService:
        provider = self.services.settings.transcription_provider
        model = (
            user.get("preferred_transcription_model")
            if user and user.get("preferred_transcription_model")
            else self.services.settings.transcription_model
        )
        _, base = provider_credentials(self.services.settings, provider)
        # A per-user model choice must not be sent to the gateway as an alias it
        # does not know, so model_for gets the final say for that provider.
        if provider == "nabugate":
            model = model_for(self.services.settings, provider, "transcription")
        return TranscriptionService(
            provider=provider,
            api_key=self._api_key_for_user(user, provider),
            model=model,
            base_url=base,
        )

    def _extraction_service_for_user(self, user: dict | None) -> ExtractionService:
        # OCR/PDF extraction uses Gemini multimodal regardless of transcription provider.
        model = (
            user.get("preferred_transcription_model")
            if user and user.get("preferred_transcription_model")
            else self.services.settings.transcription_model
        )
        return ExtractionService(provider="gemini", api_key=self._api_key_for_user(user, "gemini"), model=model)

    def _search_service_for_user(self, user: dict | None) -> SearchService:
        return SearchService(self.services.repository, self._embedding_service_for_user(user))

    def _handle_open_app(self, chat_id: int) -> None:
        """Send a button that launches the cinematic Telegram Mini App."""
        url = getattr(self.services.settings, "miniapp_url", None)
        if not url:
            self.services.api.send_message(
                chat_id=chat_id,
                text="Mini App isn't configured yet. Set MINIAPP_URL to a public HTTPS URL that serves /miniapp.",
            )
            return
        self.services.api.send_message(
            chat_id=chat_id,
            text="🚀 پنل سینمایی نوت‌بوک تلگرام را باز کن و از آرشیو خودت بپرس:",
            reply_markup=self.services.api.web_app_keyboard("باز کردن Mini App", url),
        )

    def handle_update(self, update: dict[str, object]) -> None:
        callback = update.get("callback_query")
        if isinstance(callback, dict):
            self.services.api.answer_callback_query(callback["id"])
            return

        message = update.get("message")
        if not isinstance(message, dict):
            return

        chat_id = int(message["chat"]["id"])
        sender = message.get("from") or {}
        bot_user_id = int(sender.get("id", 0))
        if not bot_user_id:
            return

        self.services.repository.upsert_bot_user(
            bot_user_id=bot_user_id,
            chat_id=chat_id,
            username=sender.get("username"),
            first_name=sender.get("first_name"),
        )

        if "contact" in message:
            self._handle_contact(chat_id, bot_user_id, message["contact"])
            return

        text = str(message.get("text", "")).strip()

        if text.startswith("/"):
            self._handle_command(chat_id, bot_user_id, text)
            return

        if self._is_forwarded(message):
            self._handle_forwarded(chat_id, bot_user_id, message)
            return

        document = message.get("document")
        if isinstance(document, dict) and self._is_backup_document(document):
            self._handle_backup_document(chat_id, bot_user_id, document)
            return

        if not text:
            return

        flow = self.services.repository.get_auth_flow(bot_user_id=bot_user_id)
        if flow:
            st = flow["status"]
            if st == "awaiting_gemini_key":
                self._handle_gemini_key(chat_id, bot_user_id, text)
            elif st == "awaiting_v_project":
                self._handle_v_project(chat_id, bot_user_id, text, flow)
            elif st == "awaiting_v_region":
                self._handle_v_region(chat_id, bot_user_id, text, flow)
            elif st == "awaiting_v_index":
                self._handle_v_index(chat_id, bot_user_id, text, flow)
            elif st == "awaiting_v_endpoint":
                self._handle_v_endpoint(chat_id, bot_user_id, text, flow)
            elif st == "awaiting_v_deployed":
                self._handle_v_deployed(chat_id, bot_user_id, text, flow)
            elif st == "awaiting_api_id":
                self._handle_api_id(chat_id, bot_user_id, text, flow)
            elif st == "awaiting_api_hash":
                self._handle_api_hash(chat_id, bot_user_id, text, flow)
            elif st == "awaiting_login_phone":
                self._handle_login_phone(chat_id, bot_user_id, text, flow)
            elif st == "awaiting_code":
                self._handle_code(chat_id, bot_user_id, text, flow)
            elif st == "awaiting_password":
                self._handle_password(chat_id, bot_user_id, text, flow)

    def _handle_command(self, chat_id: int, bot_user_id: int, text: str) -> None:
        command = text.split()[0].split("@")[0].lower()
        if command == "/start":
            self._send_welcome(chat_id)
        elif command == "/help":
            self._send_help(chat_id)
        elif command == "/app":
            self._handle_open_app(chat_id)
        elif command == "/version":
            self.services.api.send_message(chat_id=chat_id, text="Bot Version: v5.0 (Stabilized Core)")
        elif command == "/connect":
            self._begin_connect(chat_id, bot_user_id)
        elif command == "/status":
            self._handle_status(chat_id, bot_user_id)
        elif command == "/disconnect":
            self._handle_disconnect(chat_id, bot_user_id)
        elif command == "/cancel":
            self.services.repository.clear_auth_flow(bot_user_id=bot_user_id)
            self.services.api.send_message(chat_id=chat_id, text="Operation cancelled.", reply_markup=TelegramBotApi.remove_keyboard())
        elif command == "/search":
            query, source, tag = self._split_filters(text.removeprefix("/search").strip())
            self._search(chat_id, bot_user_id, query, source, tag)
        elif command == "/ask":
            query, source, tag = self._split_filters(text.removeprefix("/ask").strip())
            self._ask_brain(chat_id, bot_user_id, query, source, tag)
        elif command == "/rule":
            self._handle_rule(chat_id, bot_user_id, text.removeprefix("/rule").strip())
        elif command == "/tags":
            self._handle_tags(chat_id, bot_user_id)
        elif command == "/tag":
            self._handle_tag(chat_id, bot_user_id, text.removeprefix("/tag").strip())
        elif command == "/collection":
            self._handle_collection(chat_id, bot_user_id, text.removeprefix("/collection").strip())
        elif command == "/airules":
            self._handle_airules(chat_id, bot_user_id, text.removeprefix("/airules").strip())
        elif command == "/setarchive":
            self._handle_setarchive(chat_id, bot_user_id, text.removeprefix("/setarchive").strip())
        elif command == "/summarize":
            self._handle_summarize(chat_id, bot_user_id, text.removeprefix("/summarize").strip())
        elif command == "/digest":
            self._handle_digest(chat_id, bot_user_id, text.removeprefix("/digest").strip())
        elif command == "/podcast":
            self._handle_podcast(chat_id, bot_user_id, text.removeprefix("/podcast").strip())
        elif command == "/account":
            self._handle_account(chat_id, bot_user_id)
        elif command == "/scheduled":
            self._handle_scheduled(chat_id, bot_user_id, text.removeprefix("/scheduled").strip())
        elif command == "/llmexport":
            self._handle_llm_export(chat_id, bot_user_id, text.removeprefix("/llmexport").strip())
        elif command == "/resend":
            self._handle_resend(chat_id, bot_user_id, text.removeprefix("/resend").strip())
        elif command == "/watchdeleted":
            self._handle_watchdeleted(chat_id, bot_user_id, text.removeprefix("/watchdeleted").strip())
        elif command == "/deleted":
            self._handle_deleted(chat_id, bot_user_id, text.removeprefix("/deleted").strip())
        elif command == "/topics":
            self._handle_topics(chat_id, bot_user_id, text.removeprefix("/topics").strip())
        elif command == "/timeline":
            self._handle_timeline(chat_id, bot_user_id, text.removeprefix("/timeline").strip())
        elif command == "/join":
            self._handle_join(chat_id, bot_user_id, text.removeprefix("/join").strip())
        elif command == "/ingest":
            self._handle_ingest(chat_id, bot_user_id, text.removeprefix("/ingest").strip())
        elif command == "/import":
            self._handle_import(chat_id, bot_user_id, text.removeprefix("/import").strip())
        elif command == "/backup":
            self._handle_backup_info(chat_id)
        elif command == "/jobs":
            self._handle_jobs(chat_id, bot_user_id)
        elif command == "/canceljob":
            self._handle_cancel_job(chat_id, bot_user_id, text.removeprefix("/canceljob").strip())
        elif command == "/recent":
            self._handle_recent(chat_id, bot_user_id, text.removeprefix("/recent").strip())
        elif command == "/sources":
            self._handle_sources(chat_id, bot_user_id)
        elif command == "/delete":
            self._handle_delete(chat_id, bot_user_id, text.removeprefix("/delete").strip())
        elif command == "/export":
            self._handle_export(chat_id, bot_user_id, text.removeprefix("/export").strip())
        elif command == "/stats":
            self._handle_stats(chat_id, bot_user_id)
        else:
            self.services.api.send_message(chat_id=chat_id, text="Unknown command. Send /help to see what I can do.")

    @staticmethod
    def _split_filters(text: str) -> tuple[str, str | None, str | None]:
        """Parse a query with optional ``--source <url>`` (single token) and
        ``--tag <tag>`` (rest of line, may contain spaces) filters."""
        tokens = text.split()
        source = None
        if "--source" in tokens:
            idx = tokens.index("--source")
            if idx + 1 < len(tokens):
                source = tokens[idx + 1]
                del tokens[idx:idx + 2]
        remaining = " ".join(tokens)
        tag = None
        head, sep, tail = remaining.partition("--tag")
        if sep:
            tag = tail.strip() or None
            remaining = head
        return remaining.strip(), source, tag

    def _send_welcome(self, chat_id: int) -> None:
        self.services.api.send_message(
            chat_id=chat_id,
            text=(
                "Welcome! I'm your AI Research Assistant.\n"
                "Use /connect to link your account and configure Vertex AI.\n"
                "Forward me any message to save it to your searchable inbox.\n"
                "Send me a Telegram backup (.json/.zip) to import it — see /backup.\n"
                "Send /help to see all commands."
            ),
        )

    def _send_help(self, chat_id: int) -> None:
        help_text = (
            "<b>Available commands</b>\n"
            "/connect — link your Telegram account and configure AI\n"
            "/status — show your connection and indexing status\n"
            "/disconnect — remove your saved session and credentials\n"
            "/ingest &lt;channel_url&gt; — index a channel now (quick, inline)\n"
            "/import &lt;channel_url&gt; [limit] — queue a full, resumable import\n"
            "/backup — how to import a Telegram Desktop backup (.json/.zip)\n"
            "/jobs — show your import jobs and progress\n"
            "/canceljob &lt;id&gt; — cancel a queued/running import\n"
            "/search &lt;query&gt; [--source &lt;url&gt;] [--tag &lt;tag&gt;] — keyword/semantic search\n"
            "/ask &lt;question&gt; [--source &lt;url&gt;] [--tag &lt;tag&gt;] — ask the AI over your archive\n"
            "/summarize [--source &lt;url&gt;] [--tag &lt;tag&gt;] — summarize a source, tag, or your whole archive\n"
            "/digest [days] — AI recap of recent content (default 7 days)\n"
            "/podcast [--source &lt;url&gt;] [--tag &lt;tag&gt;] [--collection &lt;name&gt;] — generate an Audio Overview\n"
            "/topics [--source &lt;url&gt;] [--tag &lt;tag&gt;] — cluster your content into topics\n"
            "/timeline [--source &lt;url&gt;] [--tag &lt;tag&gt;] [--day] — browse your archive by date\n"
            "/export [--source &lt;url&gt;] [--tag &lt;tag&gt;] — download a Markdown export\n"
            "/recent [n] — list your latest items\n"
            "/stats — an overview of your archive\n"
            "/sources — list indexed sources\n"
            "/delete &lt;channel_url&gt; — delete a source's data\n"
            "/cancel — cancel the current flow\n\n"
            "<b>Rules &amp; tags</b>\n"
            "/rule add &lt;keyword&gt; -&gt; &lt;tag&gt; — auto-tag matching content\n"
            "/rule add-ai &lt;criterion&gt; -&gt; &lt;tag&gt; — AI-judged tag (applied on /rule apply)\n"
            "/rule list — show your rules\n"
            "/rule remove &lt;id&gt; — delete a rule\n"
            "/rule apply — re-tag existing content with current rules\n"
            "/airules on|off — auto-run AI rules on each new forward (opt-in)\n"
            "/tags — list your tags and their counts\n"
            "/tag rename &lt;old&gt; -&gt; &lt;new&gt; · /tag delete &lt;tag&gt; — manage tags\n"
            "/collection new|add|list|remove|show — group tags into notebooks\n"
            "  (then /summarize --collection &lt;name&gt; or /export --collection &lt;name&gt;)\n"
            "/setarchive &lt;@channel|off&gt; — auto-forward tagged forwards to an archive channel\n\n"
            "<b>Telegram Toolset</b> (needs /connect)\n"
            "/account — your linked account info\n"
            "/scheduled &lt;chat&gt; [cancel &lt;id&gt;] — list/cancel scheduled messages\n"
            "/llmexport &lt;chat&gt; [limit] — export a chat as an AI-friendly Markdown transcript\n"
            "/resend &lt;target&gt; &lt;text&gt; — send a message on your behalf\n"
            "/watchdeleted on|off — capture incoming messages and recover deleted ones\n"
            "/deleted [n] — show recently recovered deleted messages\n\n"
            "<b>Forwarded Inbox</b>\n"
            "Forward any message to me and I'll save its text/caption to your "
            "searchable inbox. Then use /search or /ask over it."
        )
        self.services.api.send_message(chat_id=chat_id, text=help_text)

    def _handle_status(self, chat_id: int, bot_user_id: int) -> None:
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        connected = bool(user and user.get("session_string"))
        gemini_set = bool((user and user.get("gemini_api_key")) or self.services.settings.gemini_api_key)
        vertex_ready = bool(
            (user.get("vertex_project_id") if user else None) or self.services.settings.vertex_project_id
        )
        sources_count = len(self.services.repository.list_channels(owner_id=bot_user_id))
        lines = [
            "<b>Status</b>",
            f"• Account linked: {'✅ yes' if connected else '❌ no (use /connect)'}",
            f"• AI key configured: {'✅ yes' if gemini_set else '❌ no'}",
            f"• Vertex AI Search: {'✅ configured' if vertex_ready else 'local search only'}",
            f"• Indexed sources: {sources_count}",
        ]
        if connected and user.get("phone"):
            lines.append(f"• Linked phone: {user['phone']}")
        self.services.api.send_message(chat_id=chat_id, text="\n".join(lines))

    def _handle_disconnect(self, chat_id: int, bot_user_id: int) -> None:
        self.services.repository.clear_auth_flow(bot_user_id=bot_user_id)
        removed = self.services.repository.disconnect_bot_user(bot_user_id=bot_user_id)
        if removed:
            self.services.api.send_message(
                chat_id=chat_id,
                text="Disconnected. Your saved session and credentials were removed. Use /connect to link again.",
                reply_markup=TelegramBotApi.remove_keyboard(),
            )
        else:
            self.services.api.send_message(chat_id=chat_id, text="You have no linked account to disconnect.")

    def _begin_connect(self, chat_id: int, bot_user_id: int) -> None:
        self.services.repository.upsert_auth_flow(bot_user_id=bot_user_id, chat_id=chat_id, phone="", api_id=None, api_hash=None, session_string="", phone_code_hash="", status="awaiting_phone_initial")
        self.services.api.send_message(chat_id=chat_id, text="<b>Step 1: Your Profile</b>\nPlease share your phone number to register in the system:", reply_markup=TelegramBotApi.contact_keyboard())

    def _handle_contact(self, chat_id: int, bot_user_id: int, contact: dict) -> None:
        phone = normalize_phone(str(contact.get("phone_number", "")))
        self.services.repository.save_bot_user_phone(bot_user_id=bot_user_id, phone=phone)
        self.services.repository.update_auth_flow_status(bot_user_id=bot_user_id, status="awaiting_gemini_key")

        guide_text = (
            "<b>Step 2: Vertex AI API Key</b>\n"
            "Please provide your Vertex AI API Key (starts with AQ.) or AI Studio Key (starts with AIza).\n"
            "You can get it from <a href='https://aistudio.google.com/app/apikey'>Google AI Studio</a>.\n"
            "Then send it here:"
        )

        photo_path = Path("data/media/gemini_guide.jpg")
        if photo_path.exists():
            self.services.api.send_photo(chat_id=chat_id, photo_path=photo_path, caption=guide_text)
        else:
            self.services.api.send_message(chat_id=chat_id, text=guide_text, reply_markup=TelegramBotApi.remove_keyboard())

    def _handle_gemini_key(self, chat_id: int, bot_user_id: int, text: str) -> None:
        if not (text.startswith("AIza") or text.startswith("AQ.")):
            self.services.api.send_message(chat_id=chat_id, text="Invalid Key format. It should start with 'AIza' or 'AQ.'.")
            return
        self.services.repository.update_user_gemini_key(bot_user_id=bot_user_id, api_key=text)
        self.services.repository.update_auth_flow_status(bot_user_id=bot_user_id, status="awaiting_v_project")
        self.services.api.send_message(chat_id=chat_id, text="<b>Step 3: Vertex AI Search Config</b>\nPlease enter your Google Cloud <b>Project ID</b>:")

    def _handle_v_project(self, chat_id, bot_user_id, text, flow):
        self.services.repository.upsert_auth_flow(bot_user_id=bot_user_id, chat_id=chat_id, phone=flow["phone"], api_id=None, api_hash=None, session_string="", phone_code_hash="", status="awaiting_v_region", v_project=text)
        self.services.api.send_message(chat_id=chat_id, text="Enter your <b>Region</b> (e.g., us-central1):")

    def _handle_v_region(self, chat_id, bot_user_id, text, flow):
        self.services.repository.upsert_auth_flow(bot_user_id=bot_user_id, chat_id=chat_id, phone=flow["phone"], api_id=None, api_hash=None, session_string="", phone_code_hash="", status="awaiting_v_index", v_project=flow["vertex_project_id"], v_region=text)
        self.services.api.send_message(chat_id=chat_id, text="Enter your <b>Index ID</b>:")

    def _handle_v_index(self, chat_id, bot_user_id, text, flow):
        self.services.repository.upsert_auth_flow(bot_user_id=bot_user_id, chat_id=chat_id, phone=flow["phone"], api_id=None, api_hash=None, session_string="", phone_code_hash="", status="awaiting_v_endpoint", v_project=flow["vertex_project_id"], v_region=flow["vertex_region"], v_index=text)
        self.services.api.send_message(chat_id=chat_id, text="Enter your <b>Index Endpoint ID</b>:")

    def _handle_v_endpoint(self, chat_id, bot_user_id, text, flow):
        self.services.repository.upsert_auth_flow(bot_user_id=bot_user_id, chat_id=chat_id, phone=flow["phone"], api_id=None, api_hash=None, session_string="", phone_code_hash="", status="awaiting_v_deployed", v_project=flow["vertex_project_id"], v_region=flow["vertex_region"], v_index=flow["vertex_index_id"], v_endpoint=text)
        self.services.api.send_message(chat_id=chat_id, text="Enter your <b>Deployed Index ID</b>:")

    def _handle_v_deployed(self, chat_id, bot_user_id, text, flow):
        self.services.repository.upsert_auth_flow(bot_user_id=bot_user_id, chat_id=chat_id, phone=flow["phone"], api_id=None, api_hash=None, session_string="", phone_code_hash="", status="awaiting_api_id", v_project=flow["vertex_project_id"], v_region=flow["vertex_region"], v_index=flow["vertex_index_id"], v_endpoint=flow["vertex_endpoint_id"], v_deployed=text)
        self.services.api.send_message(chat_id=chat_id, text="<b>Step 4: Telegram API</b>\nGo to <a href='https://my.telegram.org'>my.telegram.org</a> and create an app.\nSend your <b>API_ID</b>:")

    def _handle_api_id(self, chat_id: int, bot_user_id: int, text: str, flow: dict) -> None:
        if not text.isdigit():
            return
        self.services.repository.upsert_auth_flow(bot_user_id=bot_user_id, chat_id=chat_id, phone=flow["phone"], api_id=int(text), api_hash=None, session_string="", phone_code_hash="", status="awaiting_api_hash", v_project=flow["vertex_project_id"], v_region=flow["vertex_region"], v_index=flow["vertex_index_id"], v_endpoint=flow["vertex_endpoint_id"], v_deployed=flow["vertex_deployed_index_id"])
        self.services.api.send_message(chat_id=chat_id, text="Send your <b>API_HASH</b>:")

    def _handle_api_hash(self, chat_id: int, bot_user_id: int, text: str, flow: dict) -> None:
        self.services.repository.upsert_auth_flow(bot_user_id=bot_user_id, chat_id=chat_id, phone=flow["phone"], api_id=flow["api_id"], api_hash=text, session_string="", phone_code_hash="", status="awaiting_login_phone", v_project=flow["vertex_project_id"], v_region=flow["vertex_region"], v_index=flow["vertex_index_id"], v_endpoint=flow["vertex_endpoint_id"], v_deployed=flow["vertex_deployed_index_id"])
        self.services.api.send_message(chat_id=chat_id, text="<b>Step 5: Phone Number to Link</b>\nSend the phone number of the account you want to link (+98...):")

    def _handle_login_phone(self, chat_id: int, bot_user_id: int, text: str, flow: dict) -> None:
        phone = normalize_phone(text)
        if not phone:
            self.services.api.send_message(chat_id=chat_id, text="Invalid phone number.")
            return
        self.services.api.send_message(chat_id=chat_id, text="Requesting code from Telegram... (Please wait up to 1 minute)")
        try:
            res = self._async_to_sync(request_login_code(self.services.settings, phone, api_id=flow["api_id"], api_hash=flow["api_hash"]), timeout=120)
            self.services.repository.upsert_auth_flow(bot_user_id=bot_user_id, chat_id=chat_id, phone=phone, api_id=flow["api_id"], api_hash=flow["api_hash"], session_string=res["session_string"], phone_code_hash=res["phone_code_hash"], status="awaiting_code", v_project=flow["vertex_project_id"], v_region=flow["vertex_region"], v_index=flow["vertex_index_id"], v_endpoint=flow["vertex_endpoint_id"], v_deployed=flow["vertex_deployed_index_id"])
            self.services.api.send_message(chat_id=chat_id, text="Code sent to your Telegram. Enter it here:")
        except Exception as e:
            logger.exception("Auth error while requesting login code")
            self.services.api.send_message(chat_id=chat_id, text=f"Error: {e}")

    def _handle_code(self, chat_id: int, bot_user_id: int, text: str, flow: dict) -> None:
        code = normalize_code(text)
        if not code:
            self.services.api.send_message(chat_id=chat_id, text="Invalid code format. Please try again.")
            return
        self.services.api.send_message(chat_id=chat_id, text="Verifying code with Telegram...")
        try:
            res = self._async_to_sync(sign_in_with_code(self.services.settings, phone=flow["phone"], session_string=flow["session_string"], code=code, phone_code_hash=flow["phone_code_hash"], api_id=flow["api_id"], api_hash=flow["api_hash"]), timeout=60)
            if res["status"] == "password_required":
                self.services.repository.update_auth_flow_status(bot_user_id=bot_user_id, status="awaiting_password")
                self.services.api.send_message(chat_id=chat_id, text="Two-step verification enabled. Enter your password:")
                return
            self.services.repository.save_bot_user_session(bot_user_id=bot_user_id, phone=flow["phone"], api_id=flow["api_id"], api_hash=flow["api_hash"], session_string=res["session_string"], connected_at=res["connected_at"], v_project=flow["vertex_project_id"], v_region=flow["vertex_region"], v_index=flow["vertex_index_id"], v_endpoint=flow["vertex_endpoint_id"], v_deployed=flow["vertex_deployed_index_id"])
            self.services.repository.clear_auth_flow(bot_user_id=bot_user_id)
            self.services.api.send_message(chat_id=chat_id, text="Connected! Everything is ready.")
        except Exception as e:
            logger.exception("Code verification error during sign in")
            self.services.api.send_message(chat_id=chat_id, text=f"Error during sign in: {e}")

    def _handle_password(self, chat_id: int, bot_user_id: int, text: str, flow: dict) -> None:
        try:
            res = self._async_to_sync(sign_in_with_password(self.services.settings, session_string=flow["session_string"], password=text, api_id=flow["api_id"], api_hash=flow["api_hash"]))
            self.services.repository.save_bot_user_session(bot_user_id=bot_user_id, phone=flow["phone"], api_id=flow["api_id"], api_hash=flow["api_hash"], session_string=res["session_string"], connected_at=res["connected_at"], v_project=flow["vertex_project_id"], v_region=flow["vertex_region"], v_index=flow["vertex_index_id"], v_endpoint=flow["vertex_endpoint_id"], v_deployed=flow["vertex_deployed_index_id"])
            self.services.repository.clear_auth_flow(bot_user_id=bot_user_id)
            self.services.api.send_message(chat_id=chat_id, text="Connected!")
        except Exception as e:
            logger.exception("Password sign-in error")
            self.services.api.send_message(chat_id=chat_id, text=f"Error: {e}")

    def _vertex_search_config(self, user: dict | None) -> dict | None:
        """Build a Vertex AI Search config from per-user data, falling back to settings."""
        s = self.services.settings
        v_project = (user.get("vertex_project_id") if user else None) or s.vertex_project_id
        v_region = (user.get("vertex_region") if user else None) or s.vertex_region
        v_endpoint = (user.get("vertex_endpoint_id") if user else None) or s.vertex_endpoint_id
        v_deployed = (user.get("vertex_deployed_index_id") if user else None) or s.vertex_deployed_index_id
        if v_project and v_region and v_endpoint and v_deployed:
            return {
                "api_key": (user.get("gemini_api_key") if user else None) or s.gemini_api_key,
                "project_id": v_project,
                "region": v_region,
                "index_endpoint_id": v_endpoint,
                "deployed_index_id": v_deployed,
            }
        return None

    def _vertex_ingest_config(self, user: dict | None) -> dict | None:
        """Build a Vertex AI upsert config (used while indexing) from per-user data."""
        s = self.services.settings
        v_project = (user.get("vertex_project_id") if user else None) or s.vertex_project_id
        v_region = (user.get("vertex_region") if user else None) or s.vertex_region
        v_index_id = (user.get("vertex_index_id") if user else None) or s.vertex_index_id
        if v_project and v_region and v_index_id:
            return {
                "api_key": (user.get("gemini_api_key") if user else None) or s.gemini_api_key,
                "project_id": v_project,
                "region": v_region,
                "index_id": v_index_id,
            }
        return None

    @staticmethod
    def _is_forwarded(message: dict) -> bool:
        return any(
            key in message
            for key in ("forward_origin", "forward_from", "forward_from_chat", "forward_sender_name", "forward_date")
        )

    @staticmethod
    def _forward_source_label(message: dict) -> str:
        origin = message.get("forward_origin")
        if isinstance(origin, dict):
            otype = origin.get("type")
            if otype == "user":
                u = origin.get("sender_user") or {}
                name = " ".join(p for p in (u.get("first_name"), u.get("last_name")) if p)
                return name or u.get("username") or "Unknown user"
            if otype == "hidden_user":
                return origin.get("sender_user_name") or "Hidden user"
            if otype == "chat":
                c = origin.get("sender_chat") or {}
                return c.get("title") or c.get("username") or "Unknown chat"
            if otype == "channel":
                c = origin.get("chat") or {}
                return c.get("title") or c.get("username") or "Unknown channel"
        ffc = message.get("forward_from_chat")
        if isinstance(ffc, dict):
            return ffc.get("title") or ffc.get("username") or "Unknown chat"
        ff = message.get("forward_from")
        if isinstance(ff, dict):
            name = " ".join(p for p in (ff.get("first_name"), ff.get("last_name")) if p)
            return name or ff.get("username") or "Unknown user"
        if message.get("forward_sender_name"):
            return str(message["forward_sender_name"])
        return "Forwarded"

    @staticmethod
    def _forward_message_url(message: dict) -> str | None:
        """Public t.me link when the forward came from a public channel, else None."""
        origin = message.get("forward_origin")
        if isinstance(origin, dict) and origin.get("type") == "channel":
            chat = origin.get("chat") or {}
            if chat.get("username") and origin.get("message_id"):
                return f"https://t.me/{chat['username']}/{origin['message_id']}"
        ffc = message.get("forward_from_chat")
        if isinstance(ffc, dict) and ffc.get("username") and message.get("forward_from_message_id"):
            return f"https://t.me/{ffc['username']}/{message['forward_from_message_id']}"
        return None

    @staticmethod
    def _forward_media_tag(message: dict) -> str | None:
        if "photo" in message:
            return "[Forwarded photo]"
        if "document" in message:
            name = (message.get("document") or {}).get("file_name")
            return f"[Forwarded document: {name}]" if name else "[Forwarded document]"
        if "video" in message:
            return "[Forwarded video]"
        if "audio" in message:
            audio = message.get("audio") or {}
            title = audio.get("title") or audio.get("file_name")
            return f"[Forwarded audio: {title}]" if title else "[Forwarded audio]"
        if "voice" in message:
            return "[Forwarded voice message]"
        return None

    @staticmethod
    def _forward_file_ref(message: dict) -> dict | None:
        """Pick the single downloadable media file from a forwarded message."""
        if "voice" in message:
            v = message["voice"] or {}
            return {"file_id": v.get("file_id"), "kind": "voice", "file_name": "voice.ogg", "mime_type": v.get("mime_type") or "audio/ogg"}
        if "audio" in message:
            a = message["audio"] or {}
            return {"file_id": a.get("file_id"), "kind": "audio", "file_name": a.get("file_name") or "audio", "mime_type": a.get("mime_type") or "audio/mpeg"}
        if "video" in message:
            v = message["video"] or {}
            return {"file_id": v.get("file_id"), "kind": "video", "file_name": v.get("file_name") or "video.mp4", "mime_type": v.get("mime_type") or "video/mp4"}
        if "video_note" in message:
            v = message["video_note"] or {}
            return {"file_id": v.get("file_id"), "kind": "video_note", "file_name": "video_note.mp4", "mime_type": "video/mp4"}
        if "document" in message:
            d = message["document"] or {}
            return {"file_id": d.get("file_id"), "kind": "document", "file_name": d.get("file_name") or "document", "mime_type": d.get("mime_type") or ""}
        if "photo" in message:
            sizes = message.get("photo") or []
            if sizes:  # Telegram lists photo sizes ascending; take the largest
                return {"file_id": sizes[-1].get("file_id"), "kind": "photo", "file_name": "photo.jpg", "mime_type": "image/jpeg"}
        return None

    # Telegram Bot API media kinds -> the normalized kinds understood by route_media.
    _BOT_MEDIA_KINDS = {
        "voice": "audio",
        "audio": "audio",
        "video": "video",
        "video_note": "video",
        "photo": "image",
        "document": "document",
    }

    @classmethod
    def _media_route(cls, kind: str, mime_type: str | None, file_name: str | None = None) -> str | None:
        """Which processor handles a forwarded media kind: 'transcribe', 'extract', 'office', or None.

        Normalizes the Bot API kind and defers to the shared ``route_media`` so the
        forwarded inbox and full channel imports route media identically.
        """
        return route_media(cls._BOT_MEDIA_KINDS.get(kind), mime_type, file_name)

    def _process_forwarded_media(self, message: dict, *, transcription, extraction, download) -> str | None:
        """Download the forward's media and turn it into text (transcript or OCR).

        ``download(file_id, file_name)`` returns a local ``Path`` (or None). The
        transcription/extraction services and download are injected so the routing
        is fully testable without network or the event loop.
        """
        ref = self._forward_file_ref(message)
        if not ref or not ref.get("file_id"):
            return None
        route = self._media_route(ref["kind"], ref.get("mime_type"), ref.get("file_name"))
        if route is None:
            return None
        # Office docs (DOCX/XLSX) are parsed locally — no service or API key required.
        if route != "office":
            service = transcription if route == "transcribe" else extraction
            if not (service and service.enabled):
                return None
        path = download(ref["file_id"], ref["file_name"])
        if not path:
            return None
        try:
            if route == "transcribe":
                return (transcription.transcribe_media(path, path.parent) or "").strip()
            if route == "office":
                kind = detect_office_kind(ref.get("file_name"), ref.get("mime_type"))
                return (extract_office_text(path, kind) or "").strip()
            return (extraction.extract(path) or "").strip()
        except Exception:
            logger.exception("Forwarded media %s failed", route)
            return None

    def _extract_forwarded_media(self, user: dict | None, message: dict) -> str | None:
        """Wire real services + Bot API download around ``_process_forwarded_media``."""
        tmpdir = Path(tempfile.mkdtemp(prefix="fwd_media_"))

        def download(file_id: str, file_name: str) -> Path | None:
            try:
                meta = self.services.api.get_file(file_id)
                file_path = meta.get("file_path")
                if not file_path:
                    return None
                return self.services.api.download_file(str(file_path), tmpdir / (file_name or "file"))
            except Exception:
                logger.exception("Failed to download forwarded media")
                return None

        try:
            return self._process_forwarded_media(
                message,
                transcription=self._transcription_service_for_user(user),
                extraction=self._extraction_service_for_user(user),
                download=download,
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _handle_forwarded(self, chat_id: int, bot_user_id: int, message: dict) -> None:
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        label = self._forward_source_label(message)
        raw_text = str(message.get("text") or message.get("caption") or "").strip()
        media_tag = self._forward_media_tag(message)

        # Download and process any attached media (audio/video -> transcript, image/PDF -> OCR).
        file_ref = self._forward_file_ref(message)
        extracted_text = None
        if file_ref:
            self.services.api.send_message(chat_id, "Processing forwarded media…")
            extracted_text = self._extract_forwarded_media(user, message)

        body = "\n".join(part for part in (media_tag, raw_text, extracted_text) if part).strip()
        if not body:
            self.services.api.send_message(
                chat_id,
                "I couldn't extract any text from that forward yet. Media-only items "
                "(photos or files without a caption) aren't indexed in this version.",
            )
            return
        # Media was attached but produced no text (no Gemini key, or unsupported type).
        if file_ref and not extracted_text and not raw_text:
            self.services.api.send_message(
                chat_id,
                "Saved the media reference, but I couldn't read its contents. "
                "Connect a Gemini API key (via /connect) so I can transcribe audio/video and OCR images/PDFs.",
            )

        forward_key = int(message["message_id"])
        message_url = self._forward_message_url(message)
        vertex_config = self._vertex_ingest_config(user)
        tags = match_tags(body, self.services.repository.list_rules(owner_id=bot_user_id))

        async def _do() -> bool:
            pipeline = IngestionPipeline(
                settings=self.services.settings,
                repository=self.services.repository,
                transcription=self._transcription_service_for_user(user),
                embeddings=self._embedding_service_for_user(user),
                ai_classifier=self._ai_classifier_for_user(user),
            )
            return await pipeline.ingest_forwarded_message(
                owner_id=bot_user_id,
                source_label=label,
                text=body,
                forward_key=forward_key,
                message_url=message_url,
                vertex_config=vertex_config,
            )

        try:
            stored = self._async_to_sync(_do())
        except Exception as e:
            logger.exception("Forward ingest failed for user %s", bot_user_id)
            self.services.api.send_message(chat_id, f"Error saving forward: {e}")
            return

        if not stored:
            self.services.api.send_message(chat_id, "You've already saved this forward.")
            return

        prefix = "Transcribed and saved" if extracted_text else "Saved"
        reply = f"{prefix} to your inbox from “{html.escape(label)}”. Use /search or /ask to query it."
        if self._auto_forward(user=user, label=label, tags=tags, text=body, message_url=message_url):
            tag_str = ", ".join(f"#{html.escape(t)}" for t in sorted(tags))
            reply += f"\nAuto-forwarded to your archive ({tag_str})."
        self.services.api.send_message(chat_id, reply)

    def _auto_forward(self, *, user: dict | None, label: str, tags: set[str], text: str, message_url: str | None) -> bool:
        """Forward a tagged inbox item to the user's archive channel. Returns True if sent.

        Messages are sent with ``parse_mode: HTML``, so every user-controlled field
        (source label, tags, text, link) is HTML-escaped — otherwise a literal ``<``,
        ``>`` or ``&`` would break Telegram's parser and the forward would silently fail.
        """
        archive = (user or {}).get("archive_chat_id")
        if not archive or not tags:
            return False
        tag_str = " ".join(f"#{html.escape(t)}" for t in sorted(tags))
        parts = [f"📥 <b>{html.escape(label)}</b>", tag_str, html.escape(text[:1000])]
        if message_url:
            parts.append(html.escape(message_url))
        try:
            self.services.api.send_message(archive, "\n".join(p for p in parts if p))
            return True
        except Exception:
            logger.exception("Auto-forward to archive failed for user %s", user.get("bot_user_id") if user else None)
            return False

    def _make_auto_forward(self, user: dict | None):
        """Build a pipeline auto-forward callback, or None when no archive is set.

        Lets channel imports push each tagged item to the user's archive channel,
        reusing the same formatting/escaping as the forwarded-inbox path.
        """
        if not (user and user.get("archive_chat_id")):
            return None

        def _forward(label: str, tags: set[str], text: str, message_url: str | None) -> None:
            self._auto_forward(user=user, label=label, tags=tags, text=text, message_url=message_url)

        return _forward

    def _search(self, chat_id: int, bot_user_id: int, query: str, source: str | None, tag: str | None = None) -> None:
        if not query:
            self.services.api.send_message(chat_id=chat_id, text="Usage: /search <query> [--source <url>] [--tag <tag>]")
            return
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        vertex_config = self._vertex_search_config(user)
        try:
            results = self._search_service_for_user(user).search(
                owner_id=bot_user_id, query=query, channel_url=source, tag=tag, top_k=5, vertex_config=vertex_config
            )
        except Exception:
            logger.exception("Search failed for user %s", bot_user_id)
            self.services.api.send_message(chat_id=chat_id, text="Search failed. Please try again later.")
            return
        if not results:
            self.services.api.send_message(chat_id=chat_id, text="No results found.")
        else:
            resp = [f"{r.channel_title}\n{r.chunk_text[:300]}\n{r.message_url}" for r in results]
            self.services.api.send_message(chat_id=chat_id, text="\n\n".join(resp))

    def _ask_brain(self, chat_id: int, bot_user_id: int, query: str, source: str | None, tag: str | None = None) -> None:
        if not query:
            self.services.api.send_message(chat_id=chat_id, text="Usage: /ask <question> [--source <url>] [--tag <tag>]")
            return
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        vertex_config = self._vertex_search_config(user)

        self.services.api.send_message(chat_id=chat_id, text="AI Brain is thinking...")
        try:
            search_service = self._search_service_for_user(user)
            results = search_service.search(owner_id=bot_user_id, query=query, channel_url=source, tag=tag, top_k=5, vertex_config=vertex_config)
            answer = search_service.generate_answer(
                query=query,
                results=results,
                **_llm_kwargs_for(self.services.settings, user),
            )
        except Exception:
            logger.exception("Ask failed for user %s", bot_user_id)
            self.services.api.send_message(chat_id=chat_id, text="Sorry, I couldn't generate an answer right now. Please try again later.")
            return

        rendered, parse_mode = to_telegram_markdown(f"**AI Answer:**\n\n{answer}")
        self.services.api.send_message(chat_id=chat_id, text=rendered, parse_mode=parse_mode)
        if results:
            used = citations.cited_indices(answer, len(results))
            block = citations.format_sources_block(
                results, used or None, limit=None if used else 3, escape=html.escape,
            )
            self.services.api.send_message(chat_id=chat_id, text=f"<b>Sources:</b>\n{block}", disable_web_page_preview=True)

    def _handle_sources(self, chat_id: int, bot_user_id: int) -> None:
        ch = self.services.repository.list_channels(owner_id=bot_user_id)
        if not ch:
            self.services.api.send_message(chat_id, "No sources indexed.")
        else:
            self.services.api.send_message(chat_id, "\n".join([f"{c.get('channel_title') or 'Unknown'}: {c['channel_url']}" for c in ch]))

    def _handle_delete(self, chat_id: int, bot_user_id: int, link: str) -> None:
        if not link:
            self.services.api.send_message(chat_id, "Usage: /delete <channel_url>")
            return
        if self.services.repository.delete_channel_data(owner_id=bot_user_id, channel_url=link):
            self.services.api.send_message(chat_id, "Deleted.")
        else:
            self.services.api.send_message(chat_id, "Not found.")

    @staticmethod
    def _parse_rule_add(rest: str) -> tuple[str, str] | None:
        if "->" not in rest:
            return None
        keyword, _, tag = rest.partition("->")
        keyword, tag = keyword.strip(), tag.strip()
        if not keyword or not tag:
            return None
        return keyword, tag

    def _handle_rule(self, chat_id: int, bot_user_id: int, args: str) -> None:
        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""

        if sub in ("add", "add-ai"):
            parsed = self._parse_rule_add(rest)
            if not parsed:
                fmt = "criterion" if sub == "add-ai" else "keyword"
                self.services.api.send_message(chat_id, f"Usage: /rule {sub} <{fmt}> -> <tag>")
                return
            keyword, tag = parsed
            kind = "ai" if sub == "add-ai" else "keyword"
            self.services.repository.add_rule(
                owner_id=bot_user_id, keyword=keyword, tag=tag, kind=kind,
                created_at=datetime.now(UTC).isoformat(),
            )
            noun = "AI rule" if kind == "ai" else "Rule"
            self.services.api.send_message(chat_id, f"{noun} added: “{keyword}” → {tag}\nRun /rule apply to tag existing items.")
        elif sub in ("", "list"):
            rules = self.services.repository.list_rules(owner_id=bot_user_id)
            if not rules:
                self.services.api.send_message(chat_id, "No rules yet. Add one with: /rule add <keyword> -> <tag>")
                return
            lines = ["<b>Your rules</b>"] + [
                f"#{r['id']}: {'🤖' if r.get('kind') == 'ai' else '📝'} “{r['keyword']}” → {r['tag']}" for r in rules
            ]
            self.services.api.send_message(chat_id, "\n".join(lines))
        elif sub == "remove":
            try:
                rule_id = int(rest)
            except ValueError:
                self.services.api.send_message(chat_id, "Usage: /rule remove <rule_id>")
                return
            if self.services.repository.remove_rule(owner_id=bot_user_id, rule_id=rule_id):
                self.services.api.send_message(chat_id, f"Rule #{rule_id} removed. Run /rule apply to refresh tags.")
            else:
                self.services.api.send_message(chat_id, "Rule not found.")
        elif sub == "apply":
            self._retag_all(chat_id, bot_user_id)
        else:
            self.services.api.send_message(chat_id, "Usage: /rule add|list|remove|apply")

    def _retag_all(self, chat_id: int, bot_user_id: int) -> None:
        """Recompute tags for all of the user's stored content from the current rules.

        Keyword rules run locally; AI rules (if any) are evaluated with one LLM call
        per item, and are skipped with a note when no LLM backend is available.
        """
        rules = self.services.repository.list_rules(owner_id=bot_user_id)
        ai_rules = [r for r in rules if r.get("kind") == "ai"]

        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        generate = _llm_callable_for(self.services.settings, user) if ai_rules else None
        ai_active = generate is not None

        self.services.repository.clear_tags(owner_id=bot_user_id)
        tagged = 0
        for item in self.services.repository.media_texts(owner_id=bot_user_id):
            tags = match_tags(item.get("text"), rules)
            if ai_active:
                try:
                    tags |= classify_ai_tags(item.get("text") or "", ai_rules, generate=generate)
                except Exception:
                    logger.exception("AI rule classification failed for user %s", bot_user_id)
            if tags:
                self.services.repository.tag_media(owner_id=bot_user_id, media_item_id=item["media_item_id"], tags=tags)
                tagged += 1

        msg = f"Re-tagged {tagged} item(s) using {len(rules)} rule(s)."
        if ai_rules and not ai_active:
            msg += f"\nNote: {len(ai_rules)} AI rule(s) were skipped — start a local Ollama server, or connect an API key (via /connect), to enable them."
        self.services.api.send_message(chat_id, msg)

    @staticmethod
    def _extract_collection(args: str) -> tuple[str | None, str]:
        """Pull a ``--collection <name>`` flag out of ``args``; return (name, remaining)."""
        tokens = args.split()
        if "--collection" not in tokens:
            return None, args
        i = tokens.index("--collection")
        name = tokens[i + 1] if i + 1 < len(tokens) else ""
        end = i + 2 if name else i + 1
        return (name or None), " ".join(tokens[:i] + tokens[end:])

    def _collection_items(self, bot_user_id: int, name: str) -> tuple[list | None, str]:
        """Resolve a collection to its items + scope label, or (None, _) if it doesn't exist."""
        tags = self.services.repository.collection_tags(owner_id=bot_user_id, name=name)
        if tags is None:
            return None, name
        items = self.services.repository.items_for_tags(owner_id=bot_user_id, tags=tags, limit=2000)
        return items, f"collection “{name}”"

    def _handle_summarize(self, chat_id: int, bot_user_id: int, args: str) -> None:
        collection, args = self._extract_collection(args)
        if collection:
            items, scope_label = self._collection_items(bot_user_id, collection)
            if items is None:
                self.services.api.send_message(chat_id, f"Collection “{html.escape(collection)}” not found.")
                return
        else:
            _, source, tag = self._split_filters(args)
            items = self.services.repository.summary_items(owner_id=bot_user_id, channel_url=source, tag=tag)
            scope_label = tag or source or "your whole archive"
        if not items:
            self.services.api.send_message(
                chat_id,
                "Nothing to summarize yet. Ingest a channel, forward messages, or check your /sources and /tags.",
            )
            return
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)

        self.services.api.send_message(chat_id, f"Summarizing {scope_label} ({len(items)} item(s))...")
        try:
            answer = self._search_service_for_user(user).summarize(
                scope_label=scope_label,
                items=items,
                **_llm_kwargs_for(self.services.settings, user),
            )
        except Exception:
            logger.exception("Summarize failed for user %s", bot_user_id)
            self.services.api.send_message(chat_id, "Sorry, I couldn't generate a summary right now. Please try again later.")
            return
        rendered, parse_mode = to_telegram_markdown(f"**Summary — {scope_label}**\n\n{answer}")
        self.services.api.send_message(chat_id, rendered, parse_mode=parse_mode)

    def _handle_digest(self, chat_id: int, bot_user_id: int, args: str) -> None:
        days = 7
        tokens = args.split()
        if tokens:
            try:
                days = max(1, min(90, int(tokens[0])))
            except ValueError:
                pass
        since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        items = self.services.repository.recent_items(owner_id=bot_user_id, since_date=since)
        scope_label = f"the last {days} day(s)"
        if not items:
            self.services.api.send_message(chat_id, f"No new content in {scope_label}.")
            return

        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        if not _llm_available_for(self.services.settings, user):
            # Without an LLM backend, fall back to a plain count + sources digest.
            sources = sorted({
                (it.get("channel_title") or it.get("channel_url"))
                for it in items if it.get("channel_title") or it.get("channel_url")
            })
            msg = f"<b>Digest — {scope_label}</b>\n{len(items)} new item(s)."
            if sources:
                msg += "\nSources: " + ", ".join(html.escape(s) for s in sources[:10])
            self.services.api.send_message(chat_id, msg)
            return

        self.services.api.send_message(chat_id, f"Building your digest for {scope_label} ({len(items)} item(s))...")
        try:
            answer = self._search_service_for_user(user).summarize(
                scope_label=scope_label, items=items, **_llm_kwargs_for(self.services.settings, user),
            )
        except Exception:
            logger.exception("Digest failed for user %s", bot_user_id)
            self.services.api.send_message(chat_id, "Sorry, I couldn't build the digest right now. Please try again later.")
            return
        rendered, parse_mode = to_telegram_markdown(f"**Digest — {scope_label}**\n\n{answer}")
        self.services.api.send_message(chat_id, rendered, parse_mode=parse_mode)

    def _handle_podcast(self, chat_id: int, bot_user_id: int, args: str) -> None:
        """Generate an Audio Overview (podcast) from a scope of the archive."""
        collection, args = self._extract_collection(args)
        if collection:
            items, scope_label = self._collection_items(bot_user_id, collection)
            if items is None:
                self.services.api.send_message(chat_id, f"Collection “{html.escape(collection)}” not found.")
                return
        else:
            _, source, tag = self._split_filters(args)
            items = self.services.repository.summary_items(owner_id=bot_user_id, channel_url=source, tag=tag)
            scope_label = tag or source or "your whole archive"
        if not items:
            self.services.api.send_message(
                chat_id,
                "Nothing to turn into a podcast yet. Ingest a channel, forward messages, or check your /sources and /tags.",
            )
            return

        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        gemini_api_key = (user.get("gemini_api_key") if user else None) or self.services.settings.gemini_api_key
        openai_api_key = self.services.settings.openai_api_key
        if not (gemini_api_key or openai_api_key):
            self.services.api.send_message(
                chat_id,
                "An Audio Overview needs an LLM key. Connect a Gemini key via /connect (or set OPENAI_API_KEY).",
            )
            return

        self.services.api.send_message(
            chat_id,
            f"Generating an Audio Overview of {scope_label} ({len(items)} item(s))… this can take a minute.",
        )
        source_text = build_podcast_source(scope_label, items)
        audio_path: Path | None = None
        try:
            audio_path = Path(
                synthesize_podcast(
                    text=source_text,
                    gemini_api_key=gemini_api_key,
                    openai_api_key=openai_api_key,
                    tts_model=self.services.settings.podcast_tts_model,
                    llm_model_name=self.services.settings.podcast_llm_model,
                )
            )
            self.services.api.send_audio(
                chat_id=chat_id,
                audio_path=audio_path,
                caption=f"Audio Overview — {html.escape(scope_label)} ({len(items)} item(s))",
                title=f"Audio Overview — {scope_label}",
            )
        except PodcastUnavailable as exc:
            self.services.api.send_message(chat_id, str(exc))
        except Exception:
            logger.exception("Podcast generation failed for user %s", bot_user_id)
            self.services.api.send_message(chat_id, "Sorry, I couldn't generate the podcast right now. Please try again later.")
        finally:
            if audio_path is not None:
                try:
                    audio_path.unlink(missing_ok=True)
                except OSError:
                    logger.debug("Could not remove temporary podcast file %s", audio_path)

    def _handle_recent(self, chat_id: int, bot_user_id: int, args: str) -> None:
        n = 10
        tokens = args.split()
        if tokens:
            try:
                n = max(1, min(50, int(tokens[0])))
            except ValueError:
                pass
        items = self.services.repository.timeline_items(owner_id=bot_user_id, limit=n)
        if not items:
            self.services.api.send_message(chat_id, "No content yet. Ingest a channel or forward messages first.")
            return
        lines = [f"<b>Recent ({len(items)})</b>"]
        for row in recent_rows(items, limit=n):
            head = f"\n• <b>{html.escape(row['source'])}</b>"
            if row["date"]:
                head += f" · {row['date']}"
            lines.append(head)
            if row["snippet"]:
                lines.append(f"  {html.escape(row['snippet'])}")
            if row["url"]:
                lines.append(f"  {html.escape(row['url'])}")
        self.services.api.send_message(chat_id, "\n".join(lines))

    def _handle_topics(self, chat_id: int, bot_user_id: int, args: str) -> None:
        _, source, tag = self._split_filters(args)
        items = self.services.repository.chunks_with_embeddings(owner_id=bot_user_id, channel_url=source, tag=tag)
        if not items:
            self.services.api.send_message(
                chat_id,
                "No embedded content to cluster yet. Topics need embeddings — ingest with an embedding API key configured.",
            )
            return
        topics = build_topics(items, namer=self._topic_namer(bot_user_id))
        scope = tag or source or "your archive"
        lines = [f"<b>Topics — {html.escape(scope)}</b> ({len(topics)} cluster(s) from {len(items)} chunks)"]
        for topic in topics[:15]:
            lines.append(f"\n• <b>{html.escape(topic['label'])}</b> ({topic['size']})")
            if topic["sample"]:
                lines.append(f"  {html.escape(topic['sample'])}")
        self.services.api.send_message(chat_id, "\n".join(lines))

    def _topic_namer(self, bot_user_id: int):
        """Return an LLM cluster-labeller when an LLM backend is available, else None."""
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        generate = _llm_callable_for(self.services.settings, user)
        if generate is None:
            return None

        def namer(texts: list[str]) -> str:
            return label_cluster(texts, generate=generate)

        return namer

    def _ai_classifier_for_user(self, user: dict | None):
        """LLM callable for AI-rule auto-tagging, or None unless the user opted in."""
        if not (user and user.get("ai_autotag")):
            return None
        return _llm_callable_for(self.services.settings, user)

    def _handle_airules(self, chat_id: int, bot_user_id: int, args: str) -> None:
        choice = args.strip().lower()
        if choice in ("on", "off"):
            self.services.repository.set_ai_autotag(bot_user_id=bot_user_id, enabled=(choice == "on"))
            if choice == "on":
                user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
                has_key = bool((user.get("gemini_api_key") if user else None) or self.services.settings.gemini_api_key)
                note = "" if has_key else "\nNote: connect a Gemini API key (via /connect) for this to take effect."
                self.services.api.send_message(chat_id, "AI auto-tagging of new forwards is ON." + note)
            else:
                self.services.api.send_message(chat_id, "AI auto-tagging of new forwards is OFF.")
            return
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        state = "ON" if (user and user.get("ai_autotag")) else "OFF"
        self.services.api.send_message(
            chat_id,
            f"AI auto-tagging is {state}. Use /airules on|off.\n"
            "When ON, your AI rules (/rule add-ai) run automatically on each new forward "
            "(one LLM call per item). Bulk channel imports are never auto-classified.",
        )

    def _handle_timeline(self, chat_id: int, bot_user_id: int, args: str) -> None:
        granularity = "day" if "--day" in args.split() else "month"
        _, source, tag = self._split_filters(args)
        items = self.services.repository.timeline_items(owner_id=bot_user_id, channel_url=source, tag=tag)
        if not items:
            self.services.api.send_message(chat_id, "No dated content yet. Ingest a channel or forward messages first.")
            return
        periods = build_timeline(items, granularity=granularity)
        scope = tag or source or "your archive"
        lines = [f"<b>Timeline — {html.escape(scope)}</b> ({len(items)} item(s), by {granularity})"]
        for p in periods[:24]:
            sources = ", ".join(html.escape(s) for s in p["sources"][:3])
            suffix = f" — {sources}" if sources else ""
            lines.append(f"• <b>{p['period']}</b>: {p['count']} item(s){suffix}")
        self.services.api.send_message(chat_id, "\n".join(lines))

    def _handle_export(self, chat_id: int, bot_user_id: int, args: str) -> None:
        collection, args = self._extract_collection(args)
        if collection:
            items, scope = self._collection_items(bot_user_id, collection)
            if items is None:
                self.services.api.send_message(chat_id, f"Collection “{html.escape(collection)}” not found.")
                return
            slug_source = collection
        else:
            _, source, tag = self._split_filters(args)
            items = self.services.repository.summary_items(owner_id=bot_user_id, channel_url=source, tag=tag, limit=2000)
            scope = tag or source or "your whole archive"
            slug_source = tag or source or "archive"
        if not items:
            self.services.api.send_message(chat_id, "Nothing to export yet. Ingest a channel or forward messages first.")
            return
        markdown = build_markdown_export(scope, items)
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", slug_source).strip("-").lower() or "archive"
        tmpdir = Path(tempfile.mkdtemp(prefix="export_"))
        path = tmpdir / f"telegram-{slug}.md"
        path.write_text(markdown, encoding="utf-8")
        try:
            self.services.api.send_document(
                chat_id=chat_id, document_path=path,
                caption=f"Export — {html.escape(scope)} ({len(items)} item(s))",
            )
        except Exception:
            logger.exception("Export send failed for user %s", bot_user_id)
            self.services.api.send_message(chat_id, "Sorry, I couldn't send the export right now. Please try again later.")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _handle_stats(self, chat_id: int, bot_user_id: int) -> None:
        stats = self.services.repository.archive_stats(owner_id=bot_user_id)
        self.services.api.send_message(chat_id, f"<b>Archive stats</b>\n{html.escape(format_stats(stats))}")

    def _handle_tag(self, chat_id: int, bot_user_id: int, args: str) -> None:
        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""

        if sub == "rename":
            old, sep, new = rest.partition("->")
            old, new = old.strip(), new.strip()
            if not old or not new:
                self.services.api.send_message(chat_id, "Usage: /tag rename &lt;old&gt; -> &lt;new&gt;")
                return
            moved = self.services.repository.rename_tag(owner_id=bot_user_id, old_tag=old, new_tag=new)
            if moved:
                self.services.api.send_message(chat_id, f"Renamed “{html.escape(old)}” → “{html.escape(new)}” ({moved} item(s)).")
            else:
                self.services.api.send_message(chat_id, f"No items were tagged “{html.escape(old)}”.")
        elif sub == "delete":
            tag = rest.strip()
            if not tag:
                self.services.api.send_message(chat_id, "Usage: /tag delete &lt;tag&gt;")
                return
            removed = self.services.repository.delete_tag(owner_id=bot_user_id, tag=tag)
            if removed:
                self.services.api.send_message(chat_id, f"Deleted tag “{html.escape(tag)}” from {removed} item(s).")
            else:
                self.services.api.send_message(chat_id, f"No items were tagged “{html.escape(tag)}”.")
        else:
            self.services.api.send_message(chat_id, "Usage: /tag rename &lt;old&gt; -> &lt;new&gt;  ·  /tag delete &lt;tag&gt;")

    def _handle_collection(self, chat_id: int, bot_user_id: int, args: str) -> None:
        repo = self.services.repository
        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""

        if sub == "new":
            name = rest.split()[0] if rest else ""
            if not name:
                self.services.api.send_message(chat_id, "Usage: /collection new &lt;name&gt; (one word)")
                return
            repo.create_collection(owner_id=bot_user_id, name=name, created_at=datetime.now(UTC).isoformat())
            self.services.api.send_message(chat_id, f"Collection “{html.escape(name)}” created. Add tags with /collection add {html.escape(name)} &lt;tag&gt;.")
        elif sub == "add":
            toks = rest.split(maxsplit=1)
            if len(toks) < 2:
                self.services.api.send_message(chat_id, "Usage: /collection add &lt;name&gt; &lt;tag&gt;")
                return
            name, tag = toks[0], toks[1].strip()
            if repo.add_collection_tag(owner_id=bot_user_id, name=name, tag=tag):
                self.services.api.send_message(chat_id, f"Added “{html.escape(tag)}” to collection “{html.escape(name)}”.")
            else:
                self.services.api.send_message(chat_id, f"Collection “{html.escape(name)}” not found. Create it with /collection new {html.escape(name)}.")
        elif sub in ("", "list"):
            cols = repo.list_collections(owner_id=bot_user_id)
            if not cols:
                self.services.api.send_message(chat_id, "No collections yet. Create one with /collection new &lt;name&gt;.")
                return
            lines = ["<b>Your collections</b>"]
            for c in cols:
                tags = ", ".join(c["tags"]) or "—"
                lines.append(f"• <b>{html.escape(c['name'])}</b>: {html.escape(tags)}")
            self.services.api.send_message(chat_id, "\n".join(lines))
        elif sub == "remove":
            name = rest.split()[0] if rest else ""
            if not name:
                self.services.api.send_message(chat_id, "Usage: /collection remove &lt;name&gt;")
                return
            if repo.remove_collection(owner_id=bot_user_id, name=name):
                self.services.api.send_message(chat_id, f"Collection “{html.escape(name)}” removed.")
            else:
                self.services.api.send_message(chat_id, f"Collection “{html.escape(name)}” not found.")
        elif sub == "show":
            name = rest.split()[0] if rest else ""
            tags = repo.collection_tags(owner_id=bot_user_id, name=name) if name else None
            if tags is None:
                self.services.api.send_message(chat_id, f"Collection “{html.escape(name)}” not found.")
                return
            if not tags:
                self.services.api.send_message(chat_id, f"Collection “{html.escape(name)}” has no tags yet. Add some with /collection add {html.escape(name)} &lt;tag&gt;.")
                return
            items = repo.items_for_tags(owner_id=bot_user_id, tags=tags, limit=10)
            lines = [f"<b>{html.escape(name)}</b> — tags: {html.escape(', '.join(tags))} · {len(items)} recent item(s)"]
            for row in recent_rows(items, limit=10):
                head = f"\n• <b>{html.escape(row['source'])}</b>"
                if row["date"]:
                    head += f" · {row['date']}"
                lines.append(head)
                if row["snippet"]:
                    lines.append(f"  {html.escape(row['snippet'])}")
            self.services.api.send_message(chat_id, "\n".join(lines))
        else:
            self.services.api.send_message(chat_id, "Usage: /collection new|add|list|remove|show")

    def _handle_tags(self, chat_id: int, bot_user_id: int) -> None:
        tags = self.services.repository.list_tags(owner_id=bot_user_id)
        if not tags:
            self.services.api.send_message(chat_id, "No tags yet. Define rules with /rule add, then ingest or run /rule apply.")
            return
        lines = ["<b>Your tags</b>"] + [f"{t['tag']}: {t['count']}" for t in tags]
        self.services.api.send_message(chat_id, "\n".join(lines))

    def _handle_setarchive(self, chat_id: int, bot_user_id: int, args: str) -> None:
        target = args.strip()
        if not target:
            current = (self.services.repository.get_bot_user(bot_user_id=bot_user_id) or {}).get("archive_chat_id")
            if current:
                self.services.api.send_message(
                    chat_id,
                    f"Your archive is <b>{current}</b>. Tagged forwards are auto-sent there.\n"
                    "Use /setarchive off to disable, or /setarchive <@channel or chat id> to change it.",
                )
            else:
                self.services.api.send_message(
                    chat_id,
                    "Usage: /setarchive &lt;@channel or chat id&gt;\n"
                    "When set, any forward that matches a tag rule is auto-forwarded there. "
                    "Add me to that channel as an admin first. Use /setarchive off to disable.",
                )
            return
        if target.lower() in ("off", "none", "disable"):
            self.services.repository.set_archive_chat(bot_user_id=bot_user_id, archive_chat_id=None)
            self.services.api.send_message(chat_id, "Auto-forward archive disabled.")
            return
        self.services.repository.set_archive_chat(bot_user_id=bot_user_id, archive_chat_id=target)
        self.services.api.send_message(
            chat_id,
            f"Archive set to <b>{target}</b>. Forwards that match a tag rule will be auto-forwarded there.\n"
            "Make sure I'm an admin of that channel so I can post.",
        )

    # ---- Telegram-Toolset features (account / scheduled / resend / llm-export / deleted) ----

    def _build_watcher_client(self, user: dict) -> object:
        return build_client_from_session_string(
            self.services.settings, user["session_string"],
            api_id=user.get("api_id"), api_hash=user.get("api_hash"),
        )

    def _user_client(self, user: dict) -> object:
        """Build an on-demand Telethon client for a connected user."""
        return self._build_watcher_client(user)

    def _connected_user(self, chat_id: int, bot_user_id: int) -> dict | None:
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        if not user or not user.get("session_string"):
            self.services.api.send_message(chat_id, "Please /connect first.")
            return None
        return user

    def _notify_recovered(self, owner_id: int, rows: list[dict]) -> None:
        """Bot-side callback for the deleted-message watcher."""
        user = self.services.repository.get_bot_user(bot_user_id=owner_id)
        chat_id = user.get("chat_id") if user else None
        if not chat_id:
            return
        lines = [f"🗑️ Recovered {len(rows)} deleted message(s):"]
        for r in rows[:10]:
            who = r.get("sender") or "Unknown"
            text = (r.get("text") or "").strip().replace("\n", " ")
            snippet = (text[:120] + "…") if len(text) > 120 else text
            lines.append(f"• <b>{html.escape(str(who))}</b>: {html.escape(snippet)}")
        self.services.api.send_message(chat_id, "\n".join(lines))

    def _handle_account(self, chat_id: int, bot_user_id: int) -> None:
        user = self._connected_user(chat_id, bot_user_id)
        if not user:
            return
        client = self._user_client(user)

        async def _do():
            async with client:
                return await toolset.account_info(client)

        try:
            info = self._async_to_sync(_do())
        except Exception as e:
            logger.exception("Account info failed for %s", bot_user_id)
            self.services.api.send_message(chat_id, f"Error: {e}")
            return
        self.services.api.send_message(chat_id, toolset.format_account(info))

    def _handle_scheduled(self, chat_id: int, bot_user_id: int, args: str) -> None:
        user = self._connected_user(chat_id, bot_user_id)
        if not user:
            return
        tokens = args.split()
        if not tokens:
            self.services.api.send_message(chat_id, "Usage: /scheduled &lt;chat&gt; [cancel &lt;id&gt;]")
            return
        peer = tokens[0]
        client = self._user_client(user)

        if len(tokens) >= 3 and tokens[1].lower() == "cancel":
            try:
                msg_id = int(tokens[2])
            except ValueError:
                self.services.api.send_message(chat_id, "Usage: /scheduled &lt;chat&gt; cancel &lt;id&gt;")
                return

            async def _cancel():
                async with client:
                    return await toolset.cancel_scheduled(client, peer, [msg_id])

            try:
                self._async_to_sync(_cancel())
                self.services.api.send_message(chat_id, f"Cancelled scheduled message #{msg_id} in {html.escape(peer)}.")
            except Exception as e:
                logger.exception("Cancel scheduled failed for %s", bot_user_id)
                self.services.api.send_message(chat_id, f"Error: {e}")
            return

        async def _list():
            async with client:
                return await toolset.list_scheduled(client, peer)

        try:
            items = self._async_to_sync(_list())
        except Exception as e:
            logger.exception("List scheduled failed for %s", bot_user_id)
            self.services.api.send_message(chat_id, f"Error: {e}")
            return
        self.services.api.send_message(chat_id, toolset.format_scheduled(peer, items))

    def _handle_llm_export(self, chat_id: int, bot_user_id: int, args: str) -> None:
        user = self._connected_user(chat_id, bot_user_id)
        if not user:
            return
        tokens = args.split()
        if not tokens:
            self.services.api.send_message(chat_id, "Usage: /llmexport &lt;chat&gt; [limit]")
            return
        peer = tokens[0]
        limit = 200
        if len(tokens) > 1:
            try:
                limit = max(1, min(2000, int(tokens[1])))
            except ValueError:
                pass
        client = self._user_client(user)
        self.services.api.send_message(chat_id, f"Exporting up to {limit} messages from {html.escape(peer)}…")

        async def _do():
            async with client:
                return await toolset.fetch_chat_messages(client, peer, limit=limit)

        try:
            chat_label, rows = self._async_to_sync(_do(), timeout=180)
        except Exception as e:
            logger.exception("LLM export failed for %s", bot_user_id)
            self.services.api.send_message(chat_id, f"Error: {e}")
            return
        if not rows:
            self.services.api.send_message(chat_id, "No text messages found to export.")
            return
        markdown = toolset.build_llm_export(chat_label, rows)
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", peer).strip("-").lower() or "chat"
        tmpdir = Path(tempfile.mkdtemp(prefix="llmexport_"))
        path = tmpdir / f"telegram-{slug}.md"
        path.write_text(markdown, encoding="utf-8")
        try:
            self.services.api.send_document(
                chat_id=chat_id, document_path=path,
                caption=f"LLM export — {html.escape(chat_label)} ({len(rows)} msg)",
            )
        except Exception:
            logger.exception("LLM export send failed for %s", bot_user_id)
            self.services.api.send_message(chat_id, "Sorry, I couldn't send the export right now.")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _handle_resend(self, chat_id: int, bot_user_id: int, args: str) -> None:
        user = self._connected_user(chat_id, bot_user_id)
        if not user:
            return
        tokens = args.split(maxsplit=1)
        if len(tokens) < 2:
            self.services.api.send_message(chat_id, "Usage: /resend &lt;target_chat&gt; &lt;message text&gt;")
            return
        target, body = tokens[0], tokens[1]
        client = self._user_client(user)

        async def _do():
            async with client:
                return await toolset.resend_text(client, target, body)

        try:
            msg_id = self._async_to_sync(_do())
        except Exception as e:
            logger.exception("Resend failed for %s", bot_user_id)
            self.services.api.send_message(chat_id, f"Error: {e}")
            return
        self.services.api.send_message(chat_id, f"Sent to {html.escape(target)} (message #{msg_id}).")

    def _handle_watchdeleted(self, chat_id: int, bot_user_id: int, args: str) -> None:
        choice = args.strip().lower()
        if choice == "on":
            user = self._connected_user(chat_id, bot_user_id)
            if not user:
                return
            self.services.repository.set_watch_deleted(bot_user_id=bot_user_id, enabled=True)
            try:
                self.watcher.start_for_user(self.services.repository.get_bot_user(bot_user_id=bot_user_id))
            except Exception:
                logger.exception("Could not start watcher for %s", bot_user_id)
            self.services.api.send_message(
                chat_id,
                "Deleted-message watcher is ON. I'll cache new incoming messages and surface any that get deleted.\n"
                "Note: only messages received from now on can be recovered — past deletions can't.",
            )
            return
        if choice == "off":
            self.services.repository.set_watch_deleted(bot_user_id=bot_user_id, enabled=False)
            try:
                self.watcher.stop_for_user(bot_user_id)
            except Exception:
                logger.exception("Could not stop watcher for %s", bot_user_id)
            self.services.api.send_message(chat_id, "Deleted-message watcher is OFF.")
            return
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        state = "ON" if (user and user.get("watch_deleted")) else "OFF"
        self.services.api.send_message(
            chat_id,
            f"Deleted-message watcher is {state}. Use /watchdeleted on|off. See recovered messages with /deleted.",
        )

    def _handle_deleted(self, chat_id: int, bot_user_id: int, args: str) -> None:
        n = 10
        tokens = args.split()
        if tokens:
            try:
                n = max(1, min(50, int(tokens[0])))
            except ValueError:
                pass
        rows = self.services.repository.list_recovered(owner_id=bot_user_id, limit=n)
        if not rows:
            self.services.api.send_message(chat_id, "No recovered deleted messages yet. Enable capture with /watchdeleted on.")
            return
        lines = [f"<b>Recovered deleted ({len(rows)})</b>"]
        for r in rows:
            who = r.get("sender") or "Unknown"
            where = r.get("chat_title") or ""
            text = (r.get("text") or "").strip().replace("\n", " ")
            snippet = (text[:160] + "…") if len(text) > 160 else text
            head = f"\n• <b>{html.escape(str(who))}</b>"
            if where:
                head += f" · {html.escape(str(where))}"
            lines.append(head)
            lines.append(f"  {html.escape(snippet)}")
        self.services.api.send_message(chat_id, "\n".join(lines))

    def _handle_join(self, chat_id: int, bot_user_id: int, link: str) -> None:
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        if not user or not user.get("session_string"):
            self.services.api.send_message(chat_id, "Please /connect first.")
            return
        if not link:
            self.services.api.send_message(chat_id, "Usage: /join <channel_or_invite_link>")
            return
        from .telegram_client import build_client_from_session_string, join_chat
        client = build_client_from_session_string(self.services.settings, user["session_string"], api_id=user.get("api_id"), api_hash=user.get("api_hash"))

        async def _do():
            async with client:
                return await join_chat(client, link)

        try:
            self.services.api.send_message(chat_id, self._async_to_sync(_do()))
        except Exception as e:
            logger.exception("Join failed for user %s", bot_user_id)
            self.services.api.send_message(chat_id, f"Error: {e}")

    def _handle_import(self, chat_id: int, bot_user_id: int, args: str) -> None:
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        if not user or not user.get("session_string"):
            self.services.api.send_message(chat_id, "Please /connect first.")
            return
        parts = args.split()
        if not parts:
            self.services.api.send_message(chat_id, "Usage: /import <channel_url> [limit]")
            return
        url = parts[0]
        limit: int | None = None
        if len(parts) > 1:
            try:
                limit = int(parts[1])
            except ValueError:
                self.services.api.send_message(chat_id, "Limit must be a number.")
                return
        job_id = self.services.repository.create_job(
            owner_id=bot_user_id, channel_url=url, limit=limit,
            created_at=datetime.now(UTC).isoformat(),
        )
        scope = f"up to {limit}" if limit else "all"
        self.services.api.send_message(
            chat_id, f"Queued import #{job_id} for {url} ({scope} messages). Track it with /jobs."
        )

    def _handle_jobs(self, chat_id: int, bot_user_id: int) -> None:
        jobs = self.services.repository.list_jobs(owner_id=bot_user_id, limit=10)
        if not jobs:
            self.services.api.send_message(chat_id, "No import jobs yet. Start one with /import <channel_url>.")
            return
        lines = ["<b>Recent imports</b>"]
        for job in jobs:
            total = job.get("total")
            progress = f"{job.get('processed', 0)}/{total}" if total else str(job.get("processed", 0))
            line = f"#{job['id']} — {job['status']} — {job['channel_url']} ({progress})"
            if job.get("error"):
                line += f"\n  ⚠️ {str(job['error'])[:140]}"
            lines.append(line)
        lines.append("\nCancel a running import with /canceljob <id>.")
        self.services.api.send_message(chat_id, "\n".join(lines))

    def _handle_cancel_job(self, chat_id: int, bot_user_id: int, args: str) -> None:
        try:
            job_id = int(args.strip())
        except ValueError:
            self.services.api.send_message(chat_id, "Usage: /canceljob <id>")
            return
        if self.services.repository.request_job_cancel(owner_id=bot_user_id, job_id=job_id):
            self.services.api.send_message(chat_id, f"Cancellation requested for import #{job_id}. It will stop shortly.")
        else:
            self.services.api.send_message(chat_id, "No active job with that id.")

    def _run_import_job(self, job: dict, on_progress, is_cancelled) -> None:
        """Runner invoked by the JobWorker thread to execute one import job."""
        owner_id = job["owner_id"]
        user = self.services.repository.get_bot_user(bot_user_id=owner_id)
        if not user or not user.get("session_string"):
            raise RuntimeError("Account not connected. Use /connect before importing.")
        chat_id = user.get("chat_id")
        pipeline = IngestionPipeline(
            settings=self.services.settings,
            repository=self.services.repository,
            transcription=self._transcription_service_for_user(user),
            embeddings=self._embedding_service_for_user(user),
            extraction=self._extraction_service_for_user(user),
            auto_forward=self._make_auto_forward(user),
        )

        async def _do():
            return await pipeline.ingest_channel(
                owner_id=owner_id,
                channel_url=job["channel_url"],
                limit=job.get("limit_count"),
                api_id=user.get("api_id"),
                api_hash=user.get("api_hash"),
                session_string=user["session_string"],
                vertex_config=self._vertex_ingest_config(user),
                resume_from=int(job.get("cursor") or 0),
                progress_cb=on_progress,
                should_cancel=is_cancelled,
            )

        try:
            stats = asyncio.run(_do())
        except Exception as exc:
            if chat_id:
                self.services.api.send_message(chat_id, f"Import #{job['id']} failed: {exc}")
            raise

        if chat_id:
            if is_cancelled():
                self.services.api.send_message(chat_id, f"Import #{job['id']} cancelled.")
            else:
                self.services.api.send_message(
                    chat_id, f"Import #{job['id']} complete: indexed {stats.processed_media} item(s) from {stats.channel_title or stats.channel_url}."
                )

    def _handle_ingest(self, chat_id: int, bot_user_id: int, link: str) -> None:
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        if not user or not user["session_string"]:
            self.services.api.send_message(chat_id, "Please /connect first.")
            return
        if not link:
            self.services.api.send_message(chat_id, "Usage: /ingest <channel_url>")
            return

        self.services.api.send_message(chat_id, "Ingesting messages...")

        vertex_config = self._vertex_ingest_config(user)

        async def _do():
            pipeline = IngestionPipeline(
                settings=self.services.settings,
                repository=self.services.repository,
                transcription=self._transcription_service_for_user(user),
                embeddings=self._embedding_service_for_user(user),
                extraction=self._extraction_service_for_user(user),
                auto_forward=self._make_auto_forward(user),
            )
            stats = await pipeline.ingest_channel(
                owner_id=bot_user_id,
                channel_url=link,
                limit=100,
                api_id=user.get("api_id"),
                api_hash=user.get("api_hash"),
                session_string=user["session_string"],
                vertex_config=vertex_config,
            )
            return f"Indexed {stats.processed_media} items from {stats.channel_title or stats.channel_url}."

        try:
            self.services.api.send_message(chat_id, self._async_to_sync(_do()))
        except Exception as e:
            logger.exception("Ingest failed for user %s", bot_user_id)
            self.services.api.send_message(chat_id, f"Error: {e}")

    # --- Telegram Desktop backup import ------------------------------------

    @staticmethod
    def _is_backup_document(document: dict) -> bool:
        """True for an uploaded ``.json``/``.zip`` that looks like a Telegram export."""
        name = (document.get("file_name") or "").lower()
        if name.endswith((".json", ".zip")):
            return True
        mime = (document.get("mime_type") or "").lower()
        return mime in (
            "application/json",
            "application/zip",
            "application/x-zip-compressed",
            "application/x-zip",
        )

    def _handle_backup_info(self, chat_id: int) -> None:
        self.services.api.send_message(
            chat_id,
            "<b>Import a Telegram backup</b>\n"
            "In Telegram Desktop: ⋮ → <b>Export chat history</b> → format "
            "<b>Machine-readable JSON</b>. Then just send me the resulting "
            "<code>result.json</code> (or a <code>.zip</code> of the export) as a file.\n\n"
            "I'll make every message searchable (use /search and /ask) and send you "
            "back a Markdown copy. Files up to 20 MB work here; for bigger exports, "
            "use the website importer.",
        )

    def _handle_backup_document(self, chat_id: int, bot_user_id: int, document: dict) -> None:
        file_id = document.get("file_id")
        file_name = document.get("file_name") or "backup"
        size = int(document.get("file_size") or 0)
        if not file_id:
            self.services.api.send_message(chat_id, "I couldn't read that file.")
            return
        if size and size > 20 * 1024 * 1024:
            self.services.api.send_message(
                chat_id,
                "That backup is larger than 20 MB — the bot's download limit. "
                "Please import it from the website, or export a smaller date range.",
            )
            return

        self.services.api.send_message(chat_id, "Reading your Telegram backup…")
        tmpdir = Path(tempfile.mkdtemp(prefix="backup_"))
        try:
            meta = self.services.api.get_file(file_id)
            file_path = meta.get("file_path")
            if not file_path:
                self.services.api.send_message(chat_id, "I couldn't download the file from Telegram.")
                return
            local = self.services.api.download_file(str(file_path), tmpdir / file_name)
            data = local.read_bytes()
            user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
            resolver = make_zip_file_resolver(data)
            media_extractor = build_media_text_extractor(
                extraction=self._extraction_service_for_user(user),
                transcription=self._transcription_service_for_user(user),
            ) if resolver else None
            chats = parse_export(
                read_export(data, file_name), file_resolver=resolver, media_extractor=media_extractor
            )
        except Exception as exc:
            logger.exception("Backup parse failed for user %s", bot_user_id)
            self.services.api.send_message(chat_id, f"I couldn't read that backup: {exc}")
            shutil.rmtree(tmpdir, ignore_errors=True)
            return

        total = count_messages(chats)
        if not total:
            self.services.api.send_message(chat_id, "That backup had no messages I could import.")
            shutil.rmtree(tmpdir, ignore_errors=True)
            return

        self.services.api.send_message(
            chat_id, f"Importing {total} message(s) from {len(chats)} chat(s)… this can take a moment."
        )

        async def _do():
            pipeline = IngestionPipeline(
                settings=self.services.settings,
                repository=self.services.repository,
                transcription=self._transcription_service_for_user(user),
                embeddings=self._embedding_service_for_user(user),
                ai_classifier=self._ai_classifier_for_user(user),
            )
            return await pipeline.ingest_backup(
                owner_id=bot_user_id, chats=chats, vertex_config=self._vertex_ingest_config(user)
            )

        try:
            result = self._async_to_sync(_do(), timeout=1800)
        except Exception as exc:
            logger.exception("Backup ingest failed for user %s", bot_user_id)
            self.services.api.send_message(chat_id, f"Import failed: {exc}")
            shutil.rmtree(tmpdir, ignore_errors=True)
            return

        self.services.api.send_message(
            chat_id,
            f"Imported {result['messages']} message(s) from {result['chats']} chat(s) into your "
            "searchable archive. Use /search or /ask to query it.",
        )
        try:
            md_path = tmpdir / "telegram-backup.md"
            md_path.write_text(render_markdown(chats), encoding="utf-8")
            self.services.api.send_document(
                chat_id=chat_id, document_path=md_path,
                caption=f"Markdown of your backup ({total} message(s))",
            )
        except Exception:
            logger.exception("Backup markdown send failed for user %s", bot_user_id)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _handle_callback(self, callback: dict) -> None:
        pass


def main() -> None:
    setup_logging()
    bot = NotebookBot(build_services())
    bot.run_forever()


if __name__ == "__main__":
    main()
