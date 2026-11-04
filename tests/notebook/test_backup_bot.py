import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from telegram_notebook.bot import NotebookBot
from telegram_notebook.config import get_settings
from telegram_notebook.db import Repository
from telegram_notebook.telegram_backup import parse_export, read_export

SINGLE_CHAT = {
    "name": "My Channel",
    "type": "public_channel",
    "id": 999,
    "messages": [
        {"id": 1, "type": "message", "date": "2021-05-12T10:11:12", "from": "Alice", "text": "almouj townhouse price"},
        {"id": 2, "type": "service", "action": "pin_message"},
        {"id": 3, "type": "message", "date": "2021-05-12T10:13:00", "from": "Bob", "text": "second note"},
    ],
}


class _Probe(NotebookBot):
    def __init__(self):
        pass


class FakeBackupApi:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.messages = []
        self.documents = []

    def send_message(self, chat_id=None, text=None, **kwargs):
        self.messages.append((chat_id, text))

    def send_document(self, *, chat_id, document_path, caption=None):
        self.documents.append((chat_id, Path(document_path).read_text(encoding="utf-8"), caption))

    def get_file(self, file_id):
        return {"file_path": "files/result.json"}

    def download_file(self, file_path, dest):
        Path(dest).write_bytes(self.payload)
        return Path(dest)


def _probe(repo, api):
    probe = _Probe()
    probe.services = SimpleNamespace(api=api, repository=repo, settings=get_settings())
    probe.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(probe.loop)
    return probe


# --- detection -------------------------------------------------------------

def test_is_backup_document_by_extension_and_mime():
    assert NotebookBot._is_backup_document({"file_name": "result.json"})
    assert NotebookBot._is_backup_document({"file_name": "export.ZIP"})
    assert NotebookBot._is_backup_document({"mime_type": "application/zip"})
    assert not NotebookBot._is_backup_document({"file_name": "photo.jpg", "mime_type": "image/jpeg"})


# --- full handler orchestration -------------------------------------------

def test_handle_backup_document_imports_and_sends_markdown(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    payload = json.dumps(SINGLE_CHAT).encode("utf-8")
    api = FakeBackupApi(payload)
    probe = _probe(repo, api)

    probe._handle_backup_document(10, 1, {"file_id": "abc", "file_name": "result.json", "file_size": len(payload)})

    joined = " ".join(text or "" for _, text in api.messages)
    assert "Imported 2 message(s)" in joined
    # The Markdown copy is sent back as a document.
    assert len(api.documents) == 1
    assert api.documents[0][1].startswith("# Telegram Backup")
    # Content is searchable in the user's archive.
    rows = repo.keyword_candidates(owner_id=1, query="townhouse", top_k=5, channel_url=None)
    assert rows and "townhouse" in rows[0]["chunk_text"]


def test_handle_backup_document_rejects_oversized(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    api = FakeBackupApi(b"{}")
    probe = _probe(repo, api)

    probe._handle_backup_document(10, 1, {"file_id": "abc", "file_name": "big.zip", "file_size": 25 * 1024 * 1024})

    assert any("larger than 20 MB" in (t or "") for _, t in api.messages)
    assert api.documents == []


def test_handle_backup_document_reports_unreadable(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    api = FakeBackupApi(b"not json at all")
    probe = _probe(repo, api)

    probe._handle_backup_document(10, 1, {"file_id": "abc", "file_name": "result.json", "file_size": 15})

    assert any("couldn't read" in (t or "").lower() for _, t in api.messages)


# sanity: the fixtures above parse the way the handler expects
def test_fixture_parses():
    chats = parse_export(read_export(json.dumps(SINGLE_CHAT).encode("utf-8"), "result.json"))
    assert chats[0].name == "My Channel"
