from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from .chunking import split_text
from .config import Settings
from .db import Repository
from .embeddings import EmbeddingService
from .extraction import ExtractionService
from .media import route_media
from .office import detect_office_kind, extract_office_text
from .rules import classify_ai_tags, match_tags
from .transcription import TranscriptionService

if TYPE_CHECKING:
    from .telegram_backup import ParsedChat

logger = logging.getLogger(__name__)


def build_media_text_extractor(
    *,
    extraction: ExtractionService | None,
    transcription: TranscriptionService | None,
) -> Callable[[str, str, bytes], str] | None:
    """Adapt the OCR/transcription services to the backup parser's bytes callback.

    Returns a ``(route, file_name, data) -> text`` callable for ``telegram_backup``,
    or None when neither service is usable (so backups fall back to local office
    extraction only). The services operate on files, so each call writes the bytes
    to a short-lived temp file and dispatches by route.
    """
    extract_ok = bool(extraction and extraction.enabled)
    transcribe_ok = bool(transcription and transcription.enabled)
    if not (extract_ok or transcribe_ok):
        return None

    def extractor(route: str, file_name: str, data: bytes) -> str:
        if (route == "extract" and not extract_ok) or (route == "transcribe" and not transcribe_ok):
            return ""
        suffix = Path(file_name).suffix
        tmpdir = Path(tempfile.mkdtemp(prefix="backup_media_"))
        path = tmpdir / (f"media{suffix}" if suffix else "media")
        try:
            path.write_bytes(data)
            if route == "extract":
                return extraction.extract(path) or ""
            if route == "transcribe":
                return transcription.transcribe_media(path, tmpdir) or ""
            return ""
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return extractor

# auto_forward(source_label, tags, text, message_url) — push a tagged item to the
# user's archive channel. Wired only when the user has an archive configured.
AutoForward = Callable[[str, "set[str]", str, "str | None"], None]

# Synthetic source that collects messages a user forwards to the bot.
FORWARDED_INBOX_URL = "inbox://forwarded"
FORWARDED_INBOX_TITLE = "Forwarded Inbox"

# Synthetic source prefix for chats imported from a Telegram Desktop backup.
BACKUP_SOURCE_PREFIX = "backup://"


@dataclass(slots=True)
class IngestStats:
    channel_url: str
    channel_title: str | None
    processed_messages: int = 0
    processed_media: int = 0
    skipped_media: int = 0


class IngestionPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: Repository,
        transcription: TranscriptionService | None,
        embeddings: EmbeddingService,
        extraction: ExtractionService | None = None,
        ai_classifier: Callable[[str], str] | None = None,
        auto_forward: AutoForward | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.transcription = transcription
        self.embeddings = embeddings
        # OCR/PDF extractor (Gemini). When set, channel imports also turn images and
        # PDFs into searchable text; DOCX/XLSX are always parsed locally.
        self.extraction = extraction
        # When provided, AI-kind rules are evaluated with this LLM callable during
        # auto-tagging. Only the forwarded-inbox path wires it in (and only when the
        # user opted in), so bulk channel imports never trigger per-item LLM calls.
        self.ai_classifier = ai_classifier
        # When set, each newly tagged item is also forwarded to the user's archive
        # channel. Threaded through _apply_rules so it fires for channel imports too.
        self.auto_forward = auto_forward

    async def ingest_channel(
        self,
        *,
        owner_id: int,
        channel_url: str,
        limit: int | None,
        api_id: int | None = None,
        api_hash: str | None = None,
        session_string: str | None = None,
        vertex_config: dict[str, str] | None = None,
        resume_from: int = 0,
        progress_cb=None,
        should_cancel=None,
        client: object | None = None,
    ) -> IngestStats:
        from .telegram_client import (
            build_client,
            build_client_from_session_string,
            fetch_channel_info,
            iter_all_messages,
        )

        # ``client`` is injectable so the whole import path (download + extraction +
        # tagging + auto-forward) can be exercised end-to-end with a fake Telethon
        # client, with no network or real session.
        if client is None:
            if session_string is None:
                client = build_client(self.settings)
            else:
                client = build_client_from_session_string(
                    self.settings,
                    session_string,
                    api_id=api_id,
                    api_hash=api_hash,
                )
        async with client:
            channel = await fetch_channel_info(client, channel_url)
            channel_id = self.repository.upsert_channel(
                owner_id=owner_id,
                telegram_id=channel.telegram_id,
                channel_url=channel.canonical_url,
                title=channel.title,
                username=channel.username,
            )
            stats = IngestStats(
                channel_url=channel.canonical_url,
                channel_title=channel.title,
            )
            label = channel.title or channel.canonical_url

            messages = await iter_all_messages(
                client, channel_url=channel_url, limit=limit, min_id=resume_from
            )
            stats.processed_messages = len(messages)
            total = len(messages)

            for index, msg in enumerate(messages, 1):
                if should_cancel is not None and should_cancel():
                    logger.info("Ingest of %s cancelled after %s messages", channel_url, index - 1)
                    break

                message_id = self.repository.create_or_get_message(
                    channel_id=channel_id,
                    telegram_message_id=msg.telegram_message_id,
                    message_date=msg.message_date,
                    message_url=msg.message_url,
                    caption=msg.caption,
                )

                if msg.media_kind == "text":
                    if msg.caption:
                        await self._process_text_message(
                            owner_id, message_id, msg.caption, vertex_config,
                            source_label=label, message_url=msg.message_url,
                        )
                        stats.processed_media += 1
                else:
                    processed = await self._process_media_message(
                        owner_id=owner_id,
                        client=client,
                        channel_url=channel.canonical_url,
                        message_id=message_id,
                        media_message=msg,
                        vertex_config=vertex_config,
                        source_label=label,
                        message_url=msg.message_url,
                    )
                    if processed:
                        stats.processed_media += 1
                    else:
                        stats.skipped_media += 1

                # Record progress and a resume cursor (messages arrive oldest-first,
                # so the latest processed id is the highest seen so far).
                if progress_cb is not None:
                    progress_cb(index, total, msg.telegram_message_id)

            return stats

    async def ingest_forwarded_message(
        self,
        *,
        owner_id: int,
        source_label: str,
        text: str,
        forward_key: int,
        message_url: str | None = None,
        message_date: str | None = None,
        vertex_config: dict | None = None,
    ) -> bool:
        """Store a message the user forwarded to the bot into their Forwarded Inbox.

        The inbox is a per-user synthetic channel, so forwarded content becomes
        searchable through the same /search and /ask paths as ingested channels.
        Returns False when the item was already stored (idempotent on ``forward_key``).
        """
        channel_id = self.repository.upsert_channel(
            owner_id=owner_id,
            telegram_id=0,
            channel_url=FORWARDED_INBOX_URL,
            title=FORWARDED_INBOX_TITLE,
            username=None,
        )
        message_id = self.repository.create_or_get_message(
            channel_id=channel_id,
            telegram_message_id=forward_key,
            message_date=message_date or datetime.now(UTC).isoformat(),
            message_url=message_url,
            caption=source_label,
        )
        media_id = self.repository.create_or_get_media(
            message_id=message_id,
            file_name=source_label,
            file_path="",
            mime_type="text/plain",
            media_kind="forward",
            duration_seconds=None,
            file_size_bytes=len(text.encode("utf-8")),
        )
        if self.repository.media_already_transcribed(media_id):
            return False
        await self._process_text_data(media_item_id=media_id, text=text, vertex_config=vertex_config)
        self.repository.mark_media_transcribed(media_item_id=media_id, transcript_text=text)
        self._apply_rules(owner_id, media_id, text)
        return True

    async def ingest_backup(
        self,
        *,
        owner_id: int,
        chats: list[ParsedChat],
        vertex_config: dict | None = None,
    ) -> dict[str, int]:
        """Ingest parsed chats from a Telegram Desktop backup.

        Each chat becomes a synthetic ``backup://<id>`` source and every text
        message is stored, chunked, embedded and rule-tagged through the same
        path as ingested channels — so a backup is queryable via /search and
        /ask immediately. Idempotent: re-importing the same backup stores nothing
        new (messages are keyed by their in-chat id per source).
        """
        total_stored = 0
        for chat in chats:
            total_stored += await self._ingest_backup_chat(
                owner_id=owner_id, chat=chat, vertex_config=vertex_config
            )
        return {"chats": len(chats), "messages": total_stored}

    async def _ingest_backup_chat(
        self, *, owner_id: int, chat: ParsedChat, vertex_config: dict | None = None
    ) -> int:
        chat_id = chat.id
        if chat_id is None:
            channel_url = f"{BACKUP_SOURCE_PREFIX}{self._channel_storage_name(chat.name or 'chat')}"
            telegram_id = 0
        else:
            channel_url = f"{BACKUP_SOURCE_PREFIX}{chat_id}"
            telegram_id = chat_id if isinstance(chat_id, int) else 0
        channel_db_id = self.repository.upsert_channel(
            owner_id=owner_id,
            telegram_id=telegram_id,
            channel_url=channel_url,
            title=chat.name,
            username=None,
        )
        stored = 0
        for msg in chat.messages:
            text = msg.searchable_text
            if not text:
                continue
            message_id = self.repository.create_or_get_message(
                channel_id=channel_db_id,
                telegram_message_id=msg.id,
                message_date=msg.date,
                message_url=None,
                caption=msg.sender,
            )
            media_id = self.repository.create_or_get_media(
                message_id=message_id,
                file_name=msg.sender or "message",
                file_path="",
                mime_type="text/plain",
                media_kind="backup",
                duration_seconds=None,
                file_size_bytes=len(text.encode("utf-8")),
            )
            if self.repository.media_already_transcribed(media_id):
                continue
            await self._process_text_data(media_item_id=media_id, text=text, vertex_config=vertex_config)
            self.repository.mark_media_transcribed(media_item_id=media_id, transcript_text=text)
            self._apply_rules(owner_id, media_id, text)
            stored += 1
        return stored

    def _apply_rules(
        self,
        owner_id: int,
        media_item_id: int,
        text: str,
        *,
        source_label: str | None = None,
        message_url: str | None = None,
    ) -> None:
        """Auto-tag a stored item by matching the owner's rules against its text.

        Keyword rules always run; AI-kind rules only run when an ``ai_classifier``
        was supplied (the opt-in forwarded-inbox path). When a ``source_label`` is
        given and an ``auto_forward`` callback is configured, every newly tagged item
        is also pushed to the user's archive channel (used by channel imports; the
        forwarded-inbox path forwards from the bot layer instead).
        """
        rules = self.repository.list_rules(owner_id=owner_id)
        if not rules:
            return
        tags = match_tags(text, rules)
        if self.ai_classifier is not None:
            ai_rules = [r for r in rules if r.get("kind") == "ai"]
            if ai_rules:
                try:
                    tags |= classify_ai_tags(text, ai_rules, generate=self.ai_classifier)
                except Exception:
                    logger.exception("AI auto-tagging failed for owner %s", owner_id)
        if tags:
            self.repository.tag_media(owner_id=owner_id, media_item_id=media_item_id, tags=tags)
            if self.auto_forward is not None and source_label is not None:
                try:
                    self.auto_forward(source_label, tags, text, message_url)
                except Exception:
                    logger.exception("Auto-forward failed for owner %s", owner_id)

    async def _process_text_message(
        self,
        owner_id: int,
        message_id: int,
        text: str,
        vertex_config: dict | None = None,
        *,
        source_label: str | None = None,
        message_url: str | None = None,
    ) -> None:
        media_id = self.repository.create_or_get_media(
            message_id=message_id,
            file_name="text_message",
            file_path="",
            mime_type="text/plain",
            media_kind="text",
            duration_seconds=None,
            file_size_bytes=len(text.encode("utf-8")),
        )
        if self.repository.media_already_transcribed(media_id):
            return
        await self._process_text_data(media_id, text, vertex_config)
        self.repository.mark_media_transcribed(media_item_id=media_id, transcript_text=text)
        self._apply_rules(owner_id, media_id, text, source_label=source_label, message_url=message_url)

    async def _process_media_message(
        self,
        *,
        owner_id: int,
        client: object,
        channel_url: str,
        message_id: int,
        media_message: object,
        vertex_config: dict | None = None,
        source_label: str | None = None,
        message_url: str | None = None,
    ) -> bool:
        """Download a channel media item and turn it into searchable text.

        Routing mirrors the forwarded inbox: audio/video are transcribed, images and
        PDFs are OCR'd (Gemini), and DOCX/XLSX are parsed locally. When no extractor
        is available for the file (no key, or unsupported type), the message's caption
        is still indexed on its own so nothing is silently dropped.
        """
        from .telegram_client import download_message_media

        caption = (media_message.caption or "").strip()
        route = route_media(media_message.media_kind, media_message.mime_type, media_message.file_name)

        # Nothing extractable from the file itself: index the caption if present,
        # without spending a download.
        if not (route and self._route_enabled(route)):
            if not caption:
                return False
            media_id = self.repository.create_or_get_media(
                message_id=message_id,
                file_name=media_message.file_name or "media",
                file_path="",
                mime_type=media_message.mime_type,
                media_kind=media_message.media_kind,
                duration_seconds=media_message.duration_seconds,
                file_size_bytes=media_message.file_size_bytes,
            )
            if self.repository.media_already_transcribed(media_id):
                return True
            await self._process_text_data(media_item_id=media_id, text=caption, vertex_config=vertex_config)
            self.repository.mark_media_transcribed(media_item_id=media_id, transcript_text=caption)
            self._apply_rules(owner_id, media_id, caption, source_label=source_label, message_url=message_url)
            return True

        channel_name = self._channel_storage_name(channel_url)
        download_dir = self.settings.media_dir / channel_name / str(media_message.telegram_message_id)
        downloaded_path = await download_message_media(
            client,
            media_message=media_message,
            target_dir=download_dir,
        )
        if not downloaded_path:
            return False

        media_id = self.repository.create_or_get_media(
            message_id=message_id,
            file_name=media_message.file_name or downloaded_path.name,
            file_path=str(downloaded_path),
            mime_type=media_message.mime_type,
            media_kind=media_message.media_kind,
            duration_seconds=media_message.duration_seconds,
            file_size_bytes=media_message.file_size_bytes,
        )
        if self.repository.media_already_transcribed(media_id):
            return True

        extracted_text = ""
        try:
            extracted_text = self._extract_media_text(route, downloaded_path, download_dir, media_message)
        except Exception as exc:
            self.repository.mark_media_failed(media_item_id=media_id, error=str(exc))

        combined_text = self._combine_text_parts(extracted_text, media_message.caption)
        if not combined_text:
            return False

        await self._process_text_data(media_item_id=media_id, text=combined_text, vertex_config=vertex_config)
        self.repository.mark_media_transcribed(media_item_id=media_id, transcript_text=combined_text)
        self._apply_rules(owner_id, media_id, combined_text, source_label=source_label, message_url=message_url)
        return True

    def _route_enabled(self, route: str) -> bool:
        """Whether the extractor for ``route`` can run (office needs no key/service)."""
        if route == "transcribe":
            return bool(self.transcription and self.transcription.enabled)
        if route == "extract":
            return bool(self.extraction and self.extraction.enabled)
        if route == "office":
            return True
        return False

    def _extract_media_text(self, route: str, path, work_dir, media_message) -> str:
        if route == "transcribe":
            return self.transcription.transcribe_media(path, work_dir) or ""
        if route == "extract":
            return self.extraction.extract(path) or ""
        if route == "office":
            kind = detect_office_kind(media_message.file_name, media_message.mime_type)
            return extract_office_text(path, kind) or ""
        return ""

    async def _process_text_data(self, media_item_id: int, text: str, vertex_config: dict | None = None) -> None:
        chunks = split_text(
            text,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        
        datapoints = []
        indexed_chunks = []
        
        from .provider_http import vertex_ai_upsert
        
        for i, chunk in enumerate(chunks):
            v_proj = vertex_config.get("project_id") if vertex_config else None
            v_reg = vertex_config.get("region", "us-central1") if vertex_config else "us-central1"
            try:
                embedding = self.embeddings.embed(
                    chunk.text,
                    task_type="RETRIEVAL_DOCUMENT",
                    project_id=v_proj,
                    region=v_reg,
                )
            except Exception:
                logger.warning("Embedding failed for chunk %s; storing without embedding", i, exc_info=True)
                embedding = None
            indexed_chunks.append({
                "chunk_index": i,
                "text": chunk.text,
                "embedding": embedding,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            })
            if embedding:
                chunk_id = f"c_{media_item_id}_{i}"
                datapoints.append({
                    "datapoint_id": chunk_id,
                    "feature_vector": embedding
                })

        # BATCH UPSERT TO VERTEX AI
        if datapoints and vertex_config and vertex_config.get("index_id"):
            try:
                vertex_ai_upsert(
                    api_key=self.embeddings.api_key,
                    project_id=vertex_config["project_id"],
                    region=vertex_config["region"],
                    index_id=vertex_config["index_id"],
                    datapoints=datapoints
                )
            except Exception:
                logger.exception("Batch upsert to Vertex AI failed")

        self.repository.replace_chunks(media_item_id=media_item_id, chunks=indexed_chunks)

    @staticmethod
    def _combine_text_parts(transcript_text: str | None, caption: str | None) -> str:
        parts: list[str] = []
        if transcript_text and transcript_text.strip():
            parts.append(transcript_text.strip())
        if caption and caption.strip():
            caption_text = caption.strip()
            if caption_text not in parts:
                parts.append(caption_text)
        return "\n\n".join(parts).strip()

    @staticmethod
    def _channel_storage_name(channel_url: str) -> str:
        parsed = urlparse(channel_url)
        candidate = parsed.path.strip("/") or parsed.netloc or "channel"
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in candidate)
        return safe or "channel"
