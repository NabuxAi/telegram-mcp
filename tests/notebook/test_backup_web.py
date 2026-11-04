import io
import json

import telegram_notebook.main as web
from telegram_notebook.db import Repository
from telegram_notebook.embeddings import EmbeddingService
from telegram_notebook.pipeline import IngestionPipeline

SINGLE_CHAT = {
    "name": "My Channel",
    "type": "public_channel",
    "id": 555,
    "messages": [
        {"id": 1, "type": "message", "date": "2021-05-12T10:11:12", "from": "Alice", "text": "vertex search note"},
        {"id": 2, "type": "message", "date": "2021-05-12T10:13:00", "from": "Bob", "text": "second"},
    ],
}


class FakeGet(web.RequestHandler):
    def __init__(self, path):
        self.path = path
        self.client_address = ("127.0.0.1", 1)
        self.headers = {}
        self.html = None
        self.sent = None

    def _send_html(self, html, status=200):
        self.html = (status, html)

    def _send_json(self, payload, status=200):
        self.sent = (status, payload)


class FakePost(web.RequestHandler):
    def __init__(self, path, body, headers, client="127.0.0.1"):
        self.path = path
        self.rfile = io.BytesIO(body)
        self.headers = headers
        self.client_address = (client, 1)
        self.sent = None

    def _send_json(self, payload, status=200):
        self.sent = (status, payload)


# --- routing & markup ------------------------------------------------------

def test_root_serves_landing_page():
    h = FakeGet("/")
    h.do_GET()
    status, html = h.html
    assert status == 200
    assert "Telegram Notebook" in html
    assert 'href="/app"' in html


def test_app_route_serves_dashboard():
    h = FakeGet("/app")
    h.do_GET()
    _, html = h.html
    assert 'id="loadLibraryBtn"' in html  # the dashboard, not the landing page


def test_dashboard_has_backup_card():
    assert 'id="importBackupBtn"' in web.INDEX_HTML
    assert "/api/backup/import" in web.INDEX_HTML


def test_landing_has_backup_section():
    assert "Import" in web.LANDING_HTML and 'id="backup"' in web.LANDING_HTML


# --- import endpoint -------------------------------------------------------

def _wire(monkeypatch, tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    monkeypatch.setattr(web.state, "repository", repo)
    monkeypatch.setattr(web.state.settings, "web_api_token", None)  # loopback allowed
    pipeline = IngestionPipeline(
        settings=web.state.settings,
        repository=repo,
        transcription=None,
        embeddings=EmbeddingService(provider="openai", api_key=None, model="m"),
    )
    monkeypatch.setattr(web.state, "pipeline", pipeline)
    return repo


def test_backup_import_endpoint(monkeypatch, tmp_path):
    repo = _wire(monkeypatch, tmp_path)
    body = json.dumps(SINGLE_CHAT).encode("utf-8")
    headers = {"Content-Length": str(len(body)), "X-Filename": "result.json"}
    h = FakePost("/api/backup/import", body, headers)

    h.do_POST()

    status, payload = h.sent
    assert status == 200
    assert payload["ok"] is True
    assert payload["messages"] == 2
    assert payload["markdown"].startswith("# Telegram Backup")
    rows = repo.keyword_candidates(owner_id=web.WEB_OWNER_ID, query="vertex", top_k=5, channel_url=None)
    assert rows and "vertex" in rows[0]["chunk_text"]


def test_backup_import_rejects_empty_body(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    h = FakePost("/api/backup/import", b"", {"Content-Length": "0"})
    h.do_POST()
    assert h.sent[0] == 400


def test_backup_import_rejects_bad_json(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    body = b"not json"
    h = FakePost("/api/backup/import", body, {"Content-Length": str(len(body)), "X-Filename": "x.json"})
    h.do_POST()
    assert h.sent[0] == 400
