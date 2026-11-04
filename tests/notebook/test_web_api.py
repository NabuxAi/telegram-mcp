import telegram_notebook.main as web
from telegram_notebook.db import Repository


class FakeHandler(web.RequestHandler):
    """Drives do_GET without a real HTTP server."""

    def __init__(self, path, client="127.0.0.1", headers=None):
        self.path = path
        self.client_address = (client, 1)
        self.headers = headers or {}
        self.sent = None

    def _send_json(self, payload, status=200):
        self.sent = (status, payload)


def _seed(repo, *, text, date, kind="text", title="A", channel_url="https://t.me/a"):
    cid = repo.upsert_channel(owner_id=web.WEB_OWNER_ID, telegram_id=hash(channel_url) % 9999, channel_url=channel_url, title=title, username=None)
    mid = repo.create_or_get_message(channel_id=cid, telegram_message_id=hash(text) % 9999, message_date=date, message_url="u", caption=None)
    media_id = repo.create_or_get_media(message_id=mid, file_name="t", file_path="", mime_type="text/plain", media_kind=kind, duration_seconds=None, file_size_bytes=1)
    repo.mark_media_transcribed(media_item_id=media_id, transcript_text=text)


def _wire(monkeypatch, tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    monkeypatch.setattr(web.state, "repository", repo)
    monkeypatch.setattr(web.state.settings, "web_api_token", None)  # loopback allowed
    return repo


def test_api_stats(monkeypatch, tmp_path):
    repo = _wire(monkeypatch, tmp_path)
    _seed(repo, text="one", date="2026-01-01T00:00:00", kind="text")
    _seed(repo, text="two", date="2026-05-01T00:00:00", kind="voice")
    h = FakeHandler("/api/stats")
    h.do_GET()
    status, payload = h.sent
    assert status == 200
    assert payload["items"] == 2
    assert payload["by_kind"] == {"text": 1, "voice": 1}


def test_api_recent(monkeypatch, tmp_path):
    repo = _wire(monkeypatch, tmp_path)
    _seed(repo, text="newest", date="2026-06-10T00:00:00")
    _seed(repo, text="older", date="2026-06-01T00:00:00")
    h = FakeHandler("/api/recent?limit=1")
    h.do_GET()
    status, payload = h.sent
    assert status == 200
    assert len(payload["items"]) == 1 and payload["items"][0]["snippet"] == "newest"


def test_api_timeline(monkeypatch, tmp_path):
    repo = _wire(monkeypatch, tmp_path)
    _seed(repo, text="a", date="2026-06-01T00:00:00")
    _seed(repo, text="b", date="2026-04-01T00:00:00")
    h = FakeHandler("/api/timeline?granularity=month")
    h.do_GET()
    status, payload = h.sent
    assert status == 200
    assert payload["granularity"] == "month"
    assert [p["period"] for p in payload["periods"]] == ["2026-06", "2026-04"]


def test_api_stats_requires_auth_off_loopback(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    monkeypatch.setattr(web.state.settings, "web_api_token", "s3cret")
    h = FakeHandler("/api/stats", client="8.8.8.8", headers={})
    h.do_GET()
    assert h.sent[0] == 401


def test_index_html_has_library_panel():
    # The dashboard surfaces the read-only endpoints in a Library panel,
    # including the monthly timeline.
    assert 'id="loadLibraryBtn"' in web.INDEX_HTML
    assert "/api/stats" in web.INDEX_HTML
    assert "/api/recent" in web.INDEX_HTML
    assert "/api/timeline" in web.INDEX_HTML
    assert 'id="libraryTimeline"' in web.INDEX_HTML


def test_index_html_renders_citation_numbers():
    # The Ask panel numbers each source and marks the ones the answer cited.
    assert "data.cited" in web.INDEX_HTML
    assert "[${n}]" in web.INDEX_HTML
