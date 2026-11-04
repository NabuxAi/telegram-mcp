"""Parse a Telegram Desktop "Export chat history" backup into searchable items.

Telegram Desktop can export a chat (or a whole account) as *Machine-readable
JSON*. A single-chat export is a JSON object with a top-level ``messages`` list;
a full-account export nests one such object per chat under ``chats.list``. Users
often zip the export folder (the JSON plus media subfolders), so this module also
knows how to pull ``result.json`` out of a ``.zip``.

Everything here is a pure function over bytes/dicts (only the standard library),
so the parser and the Markdown renderer are trivially unit-testable, and the bot
and web layers just feed the parsed chats into the existing ingestion pipeline.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .media import route_media
from .office import detect_office_kind, extract_office_bytes

logger = logging.getLogger(__name__)

ZIP_MAGIC = b"PK\x03\x04"

# Resolve an export-relative attachment path (e.g. "files/report.docx") to its bytes.
FileResolver = Callable[[str], "bytes | None"]

# Extract text from a non-office attachment given (route, file_name, data), where
# route is "extract" (OCR/PDF) or "transcribe" (audio/video). Injected by the caller
# that owns the Gemini/transcription services; office docs are handled locally.
MediaExtractor = Callable[[str, str, bytes], str]


@dataclass(slots=True)
class ParsedMessage:
    id: int
    date: str | None
    sender: str | None
    text: str
    media_label: str | None = None
    forwarded_from: str | None = None
    reply_to: int | None = None
    document_text: str = ""  # text extracted from an attached DOCX/XLSX, if any

    @property
    def searchable_text(self) -> str:
        """Media tag + text + extracted document text, mirroring the Forwarded Inbox."""
        return "\n".join(
            part for part in (self.media_label, self.text, self.document_text) if part
        ).strip()


@dataclass(slots=True)
class ParsedChat:
    name: str
    type: str | None
    id: object
    messages: list[ParsedMessage] = field(default_factory=list)


# --- reading the upload (zip or raw json) ----------------------------------


def read_export(data: bytes, filename: str | None = None) -> dict:
    """Return the export's JSON object from raw upload ``data``.

    Handles both a raw ``result.json`` and a ``.zip`` of the export folder
    (detected by the zip magic bytes or a ``.zip`` filename).
    """
    if not data:
        raise ValueError("The uploaded file is empty.")
    is_zip = data[:4] == ZIP_MAGIC or (filename or "").lower().endswith(".zip")
    raw = _read_result_json_from_zip(data) if is_zip else data
    text = raw.decode("utf-8-sig", errors="replace")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Not valid JSON ({exc}).") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Telegram export JSON must be an object.")
    return parsed


def _read_result_json_from_zip(data: bytes) -> bytes:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"The file looks like a zip but could not be opened ({exc}).") from exc
    with archive as zf:
        names = zf.namelist()
        candidates = [n for n in names if n.rsplit("/", 1)[-1].lower() == "result.json"]
        if not candidates:
            candidates = [n for n in names if n.lower().endswith(".json")]
        if not candidates:
            raise ValueError(
                "No result.json found in the zip. Export your chat as "
                "'Machine-readable JSON' in Telegram Desktop before zipping it."
            )
        # Prefer the shallowest / shortest path (the top-level result.json).
        candidates.sort(key=lambda n: (n.count("/"), len(n)))
        return zf.read(candidates[0])


# --- parsing ---------------------------------------------------------------


def parse_export(
    data: dict,
    *,
    file_resolver: FileResolver | None = None,
    media_extractor: MediaExtractor | None = None,
) -> list[ParsedChat]:
    """Normalize a single-chat or full-account export into ``ParsedChat`` list.

    When ``file_resolver`` is supplied (e.g. built from the export zip via
    ``make_zip_file_resolver``), attached files are turned into searchable text:
    DOCX/XLSX are parsed locally, and — if a ``media_extractor`` is also given —
    images/PDFs are OCR'd and audio/video are transcribed through it.
    """
    if not isinstance(data, dict):
        raise ValueError("Telegram export must be a JSON object.")
    if isinstance(data.get("messages"), list):
        return [_parse_chat(data, file_resolver, media_extractor)]
    chats = data.get("chats")
    if isinstance(chats, dict) and isinstance(chats.get("list"), list):
        return [_parse_chat(c, file_resolver, media_extractor) for c in chats["list"] if isinstance(c, dict)]
    if isinstance(chats, list):
        return [_parse_chat(c, file_resolver, media_extractor) for c in chats if isinstance(c, dict)]
    raise ValueError(
        "Unrecognized Telegram export. Expected a chat export (with 'messages') "
        "or a full export (with 'chats.list'). Export as 'Machine-readable JSON' "
        "from Telegram Desktop."
    )


def _parse_chat(
    raw: dict,
    file_resolver: FileResolver | None = None,
    media_extractor: MediaExtractor | None = None,
) -> ParsedChat:
    messages: list[ParsedMessage] = []
    for item in raw.get("messages", []) or []:
        parsed = _parse_message(item, file_resolver, media_extractor)
        if parsed is not None:
            messages.append(parsed)
    name = (raw.get("name") or "").strip() or "Telegram chat"
    return ParsedChat(name=name, type=raw.get("type"), id=raw.get("id"), messages=messages)


def _parse_message(
    raw: object,
    file_resolver: FileResolver | None = None,
    media_extractor: MediaExtractor | None = None,
) -> ParsedMessage | None:
    if not isinstance(raw, dict):
        return None
    if raw.get("type") == "service":  # joined/pinned/etc. — no real content to index
        return None
    try:
        msg_id = int(raw.get("id"))
    except (TypeError, ValueError):
        return None
    reply_to = raw.get("reply_to_message_id")
    return ParsedMessage(
        id=msg_id,
        date=raw.get("date") or None,
        sender=(raw.get("from") or None),
        text=message_text(raw),
        media_label=media_label(raw),
        forwarded_from=(raw.get("forwarded_from") or None),
        reply_to=reply_to if isinstance(reply_to, int) else None,
        document_text=_attached_media_text(raw, file_resolver, media_extractor),
    )


def _attachment(raw: dict) -> tuple[str, str, str | None, str] | None:
    """Return ``(path, file_name, mime_type, media_kind)`` for the message's
    attachment, or None. ``media_kind`` is the normalized kind ``route_media``
    understands (image/audio/video/document); stickers are skipped.
    """
    photo = raw.get("photo")
    if isinstance(photo, str) and photo:
        return (photo, raw.get("file_name") or "photo.jpg", "image/jpeg", "image")
    file_path = raw.get("file")
    if not isinstance(file_path, str) or not file_path:
        return None
    name = raw.get("file_name") or file_path.rsplit("/", 1)[-1]
    mime = raw.get("mime_type")
    media_type = raw.get("media_type")
    if media_type == "sticker":
        return None
    if media_type in ("voice_message", "audio_file"):
        kind = "audio"
    elif media_type in ("video_message", "video_file", "animation"):
        kind = "video"
    else:
        kind = "document"
    return (file_path, name, mime, kind)


def _attached_media_text(
    raw: dict,
    file_resolver: FileResolver | None,
    media_extractor: MediaExtractor | None,
) -> str:
    """Turn an attached file into searchable text, when the export bundled it.

    DOCX/XLSX are parsed locally (no key). Images/PDFs (``extract``) and audio/video
    (``transcribe``) are routed through the injected ``media_extractor``, so they
    only run when the caller wired up the Gemini/transcription services.
    """
    if file_resolver is None:
        return ""
    attachment = _attachment(raw)
    if attachment is None:
        return ""
    path, name, mime, kind = attachment

    office_kind = detect_office_kind(name, mime)
    if office_kind:
        blob = file_resolver(path)
        if not blob:
            return ""
        try:
            return extract_office_bytes(blob, office_kind).strip()
        except Exception:
            logger.exception("Failed to extract backup document %s", path)
            return ""

    if media_extractor is None:
        return ""
    route = route_media(kind, mime, name)
    if route not in ("extract", "transcribe"):
        return ""
    blob = file_resolver(path)
    if not blob:
        return ""
    try:
        return (media_extractor(route, name, blob) or "").strip()
    except Exception:
        logger.exception("Failed to extract backup media %s", path)
        return ""


def make_zip_file_resolver(data: bytes) -> FileResolver | None:
    """Build a resolver that reads attachment bytes out of an export ``.zip``.

    Returns None when ``data`` is not a zip. Export JSON references files by a path
    relative to the export root (e.g. ``files/doc.docx``), but a zipped folder may
    prefix entries with the folder name, so matching falls back to suffix/basename.
    """
    if data[:4] != ZIP_MAGIC:
        return None
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return None
    names = archive.namelist()

    def resolve(rel_path: str) -> bytes | None:
        rel = rel_path.replace("\\", "/").lstrip("/")
        for name in names:
            if name == rel or name.endswith("/" + rel):
                return _safe_read(archive, name)
        base = rel.rsplit("/", 1)[-1]
        for name in names:
            if name.rsplit("/", 1)[-1] == base:
                return _safe_read(archive, name)
        return None

    return resolve


def _safe_read(archive: zipfile.ZipFile, name: str) -> bytes | None:
    try:
        return archive.read(name)
    except (KeyError, zipfile.BadZipFile):
        return None


def message_text(raw: dict) -> str:
    """Flatten a message's text, preferring ``text_entities`` when present.

    Telegram stores text either as a plain string or as a list of fragments
    (strings and ``{"type", "text", "href"}`` entity dicts) that concatenate to
    the full message. For ``text_link`` entities the visible text and the URL
    differ, so we inline the URL to keep links searchable.
    """
    entities = raw.get("text_entities")
    if isinstance(entities, list) and entities:
        flattened = _entities_to_text(entities).strip()
        if flattened:
            return flattened
    value = raw.get("text")
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return _entities_to_text(value).strip()
    return ""


def _entities_to_text(entities: list) -> str:
    parts: list[str] = []
    for entity in entities:
        if isinstance(entity, str):
            parts.append(entity)
        elif isinstance(entity, dict):
            text = str(entity.get("text") or "")
            href = entity.get("href")
            if entity.get("type") == "text_link" and href and href not in text:
                parts.append(f"{text} ({href})")
            else:
                parts.append(text)
    return "".join(parts)


def media_label(raw: dict) -> str | None:
    """A short ``[kind]`` tag for a message's attachment, or None for text-only."""
    if raw.get("photo"):
        return "[photo]"
    media_type = raw.get("media_type")
    if media_type:
        if media_type == "sticker":
            emoji = raw.get("sticker_emoji")
            return f"[sticker {emoji}]" if emoji else "[sticker]"
        names = {
            "voice_message": "voice message",
            "video_message": "video message",
            "video_file": "video",
            "audio_file": "audio",
            "animation": "GIF",
        }
        base = names.get(media_type, str(media_type).replace("_", " "))
        if media_type == "audio_file":
            title = raw.get("title") or raw.get("file_name")
            return f"[audio: {title}]" if title else "[audio]"
        duration = raw.get("duration_seconds")
        if media_type in ("voice_message", "video_message", "video_file") and duration:
            return f"[{base}: {duration}s]"
        return f"[{base}]"
    if raw.get("file"):
        return f"[file: {raw.get('file_name') or 'document'}]"
    if raw.get("poll"):
        question = (raw["poll"] or {}).get("question")
        return f"[poll: {question}]" if question else "[poll]"
    if raw.get("location_information"):
        return "[location]"
    if raw.get("contact_information"):
        return "[contact]"
    return None


# --- markdown rendering ----------------------------------------------------


def count_messages(chats: list[ParsedChat]) -> int:
    return sum(len(chat.messages) for chat in chats)


def render_markdown(chats: list[ParsedChat], *, generated_at: str | None = None) -> str:
    """Render parsed chats as a single, readable Markdown chat log."""
    stamp = generated_at or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    total = count_messages(chats)
    lines = [
        f"# Telegram Backup — {len(chats)} chat(s)",
        "",
        f"_Converted {stamp} · {total} message(s)_",
        "",
    ]
    for chat in chats:
        lines.append(f"## {chat.name}")
        lines.append("")
        for msg in chat.messages:
            header = []
            if msg.sender:
                header.append(f"**{msg.sender}**")
            if msg.date:
                header.append(msg.date.replace("T", " "))
            if header:
                lines.append(" · ".join(header))
            if msg.forwarded_from:
                lines.append(f"_forwarded from {msg.forwarded_from}_")
            if msg.media_label:
                lines.append(msg.media_label)
            if msg.text:
                lines.append(msg.text)
            if msg.document_text:
                lines.append(f"> {msg.document_text.replace(chr(10), chr(10) + '> ')}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
