from pathlib import Path
from types import SimpleNamespace

from telegram_notebook.bot import NotebookBot
from telegram_notebook.db import Repository
from telegram_notebook.export import build_markdown_export


class _Probe(NotebookBot):
    def __init__(self):
        pass


class FakeApi:
    def __init__(self):
        self.messages = []
        self.documents = []

    def send_message(self, chat_id=None, text=None, **kwargs):
        self.messages.append((chat_id, text))

    def send_document(self, *, chat_id, document_path, caption=None):
        self.documents.append((chat_id, Path(document_path).read_text(encoding="utf-8"), caption))


# --- pure markdown builder ---

def test_build_markdown_export_structure():
    items = [
        {"text": "first note", "channel_title": "AI News", "message_url": "https://t.me/ai/1"},
        {"text": "second", "channel_url": "https://t.me/b"},
    ]
    md = build_markdown_export("your whole archive", items, generated_at="2026-06-12 18:00 UTC")
    assert md.startswith("# Telegram Archive — your whole archive")
    assert "2 item(s)" in md
    assert "## 1. AI News" in md and "## 2. https://t.me/b" in md
    assert "[https://t.me/ai/1](https://t.me/ai/1)" in md
    assert "first note" in md and "second" in md
    assert md.endswith("\n")


def test_build_markdown_export_handles_missing_fields():
    md = build_markdown_export("tag", [{"text": ""}])
    assert "## 1. Unknown source" in md  # no source/url/text still renders a heading


# --- /export handler orchestration ---

def _seed(repo, *, owner_id, text):
    cid = repo.upsert_channel(owner_id=owner_id, telegram_id=1, channel_url="https://t.me/c", title="C", username=None)
    mid = repo.create_or_get_message(channel_id=cid, telegram_message_id=hash(text) % 9999, message_date="2026-06-01T00:00:00", message_url="https://t.me/c/1", caption=None)
    media_id = repo.create_or_get_media(message_id=mid, file_name="t", file_path="", mime_type="text/plain", media_kind="text", duration_seconds=None, file_size_bytes=1)
    repo.mark_media_transcribed(media_item_id=media_id, transcript_text=text)


def test_export_handler_sends_markdown_document(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    _seed(repo, owner_id=1, text="exportable content")
    api = FakeApi()
    probe = _Probe()
    probe.services = SimpleNamespace(repository=repo, api=api)

    probe._handle_export(10, 1, "")

    assert len(api.documents) == 1
    chat_id, content, caption = api.documents[0]
    assert chat_id == 10
    assert "exportable content" in content and content.startswith("# Telegram Archive")
    assert "Export —" in caption


def test_export_handler_empty_archive(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    api = FakeApi()
    probe = _Probe()
    probe.services = SimpleNamespace(repository=repo, api=api)

    probe._handle_export(10, 1, "")

    assert api.documents == []
    assert "Nothing to export" in api.messages[-1][1]
