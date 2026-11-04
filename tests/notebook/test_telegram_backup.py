import asyncio
import io
import json
import zipfile

import pytest

from telegram_notebook.config import get_settings
from telegram_notebook.db import Repository
from telegram_notebook.embeddings import EmbeddingService
from telegram_notebook.pipeline import IngestionPipeline
from telegram_notebook.telegram_backup import (
    count_messages,
    media_label,
    message_text,
    parse_export,
    read_export,
    render_markdown,
)

SINGLE_CHAT = {
    "name": "My Channel",
    "type": "public_channel",
    "id": 123456,
    "messages": [
        {
            "id": 1,
            "type": "message",
            "date": "2021-05-12T10:11:12",
            "from": "Alice",
            "text": "hello world",
            "text_entities": [{"type": "plain", "text": "hello world"}],
        },
        {"id": 2, "type": "service", "date": "2021-05-12T10:12:00", "action": "pin_message"},
        {
            "id": 3,
            "type": "message",
            "date": "2021-05-12T10:13:00",
            "from": "Bob",
            "text": [
                {"type": "plain", "text": "check "},
                {"type": "text_link", "text": "this", "href": "https://example.com"},
            ],
        },
        {
            "id": 4,
            "type": "message",
            "date": "2021-05-12T10:14:00",
            "from": "Bob",
            "photo": "photos/1.jpg",
            "text": "",
        },
    ],
}

FULL_EXPORT = {
    "about": "exported data",
    "chats": {
        "about": "chats",
        "list": [
            SINGLE_CHAT,
            {"name": "Notes", "type": "saved_messages", "id": 777, "messages": [
                {"id": 1, "type": "message", "date": "2022-01-01T00:00:00", "from": "Me", "text": "remember this"},
            ]},
        ],
    },
}


# --- text flattening -------------------------------------------------------

def test_message_text_plain_string():
    assert message_text({"text": "hi there"}) == "hi there"


def test_message_text_prefers_entities_and_inlines_links():
    raw = SINGLE_CHAT["messages"][2]
    assert message_text(raw) == "check this (https://example.com)"


def test_message_text_empty():
    assert message_text({"text": ""}) == ""
    assert message_text({}) == ""


# --- media labels ----------------------------------------------------------

def test_media_label_variants():
    assert media_label({"photo": "p.jpg"}) == "[photo]"
    assert media_label({"media_type": "voice_message", "duration_seconds": 12}) == "[voice message: 12s]"
    assert media_label({"media_type": "sticker", "sticker_emoji": "🔥"}) == "[sticker 🔥]"
    assert media_label({"media_type": "audio_file", "title": "song"}) == "[audio: song]"
    assert media_label({"file": "x", "file_name": "report.pdf"}) == "[file: report.pdf]"
    assert media_label({"text": "just text"}) is None


# --- parsing ---------------------------------------------------------------

def test_parse_single_chat_skips_service_messages():
    chats = parse_export(SINGLE_CHAT)
    assert len(chats) == 1
    chat = chats[0]
    assert chat.name == "My Channel" and chat.id == 123456
    assert [m.id for m in chat.messages] == [1, 3, 4]  # service id 2 dropped


def test_parse_media_only_message_is_searchable_via_label():
    chat = parse_export(SINGLE_CHAT)[0]
    photo_msg = next(m for m in chat.messages if m.id == 4)
    assert photo_msg.text == ""
    assert photo_msg.searchable_text == "[photo]"


def test_parse_full_export_lists_all_chats():
    chats = parse_export(FULL_EXPORT)
    assert {c.name for c in chats} == {"My Channel", "Notes"}
    assert count_messages(chats) == 4  # 3 from My Channel + 1 from Notes


def test_parse_unrecognized_structure_raises():
    with pytest.raises(ValueError):
        parse_export({"something": "else"})


# --- reading uploads (json / zip) -----------------------------------------

def test_read_export_from_json_bytes():
    data = json.dumps(SINGLE_CHAT).encode("utf-8")
    assert read_export(data, "result.json")["id"] == 123456


def test_read_export_from_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ChatExport/result.json", json.dumps(SINGLE_CHAT))
        zf.writestr("ChatExport/photos/1.jpg", b"\xff\xd8stub")
    parsed = read_export(buf.getvalue(), "export.zip")
    assert parsed["name"] == "My Channel"


def test_read_export_empty_raises():
    with pytest.raises(ValueError):
        read_export(b"", "x.json")


def test_read_export_zip_without_json_raises():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("photos/1.jpg", b"data")
    with pytest.raises(ValueError):
        read_export(buf.getvalue(), "export.zip")


# --- markdown rendering ----------------------------------------------------

def test_render_markdown_structure():
    md = render_markdown(parse_export(SINGLE_CHAT), generated_at="2026-06-17 10:00 UTC")
    assert md.startswith("# Telegram Backup — 1 chat(s)")
    assert "3 message(s)" in md
    assert "## My Channel" in md
    assert "**Alice**" in md and "hello world" in md
    assert "[photo]" in md
    assert md.endswith("\n")


# --- pipeline ingestion ----------------------------------------------------

def _pipeline(repo):
    return IngestionPipeline(
        settings=get_settings(),
        repository=repo,
        transcription=None,
        embeddings=EmbeddingService(provider="openai", api_key=None, model="text-embedding-3-small"),
    )


def test_ingest_backup_makes_content_searchable(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    chats = parse_export(SINGLE_CHAT)

    result = asyncio.run(_pipeline(repo).ingest_backup(owner_id=1, chats=chats))
    assert result["chats"] == 1
    assert result["messages"] == 3  # 3 text/media items stored

    rows = repo.keyword_candidates(owner_id=1, query="hello", top_k=5, channel_url=None)
    assert rows and "hello world" in rows[0]["chunk_text"]

    channels = repo.list_channels(owner_id=1)
    assert channels[0]["channel_url"] == "backup://123456"


def test_ingest_backup_is_idempotent(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    chats = parse_export(SINGLE_CHAT)

    first = asyncio.run(_pipeline(repo).ingest_backup(owner_id=1, chats=chats))
    second = asyncio.run(_pipeline(repo).ingest_backup(owner_id=1, chats=chats))
    assert first["messages"] == 3
    assert second["messages"] == 0  # re-import stores nothing new


def test_ingest_backup_isolated_per_owner(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    asyncio.run(_pipeline(repo).ingest_backup(owner_id=1, chats=parse_export(SINGLE_CHAT)))
    assert repo.keyword_candidates(owner_id=2, query="hello", top_k=5, channel_url=None) == []
