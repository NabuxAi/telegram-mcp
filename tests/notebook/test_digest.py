from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from telegram_notebook.bot import NotebookBot
from telegram_notebook.db import Repository


class _Probe(NotebookBot):
    def __init__(self):
        pass


class FakeApi:
    def __init__(self):
        self.messages = []

    def send_message(self, chat_id=None, text=None, **kwargs):
        self.messages.append((chat_id, text))


def _seed(repo, *, owner_id, text, date, channel_url="https://t.me/a", title="A"):
    cid = repo.upsert_channel(owner_id=owner_id, telegram_id=hash(channel_url) % 9999, channel_url=channel_url, title=title, username=None)
    mid = repo.create_or_get_message(channel_id=cid, telegram_message_id=hash(text) % 9999, message_date=date, message_url="u", caption=None)
    media_id = repo.create_or_get_media(message_id=mid, file_name="t", file_path="", mime_type="text/plain", media_kind="text", duration_seconds=None, file_size_bytes=1)
    repo.mark_media_transcribed(media_item_id=media_id, transcript_text=text)


def _iso(days_ago):
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()


def test_recent_items_filters_by_date_and_owner(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    _seed(repo, owner_id=1, text="fresh", date=_iso(1))
    _seed(repo, owner_id=1, text="old", date=_iso(30))
    _seed(repo, owner_id=2, text="other", date=_iso(1))

    rows = repo.recent_items(owner_id=1, since_date=_iso(7))
    assert [r["text"] for r in rows] == ["fresh"]  # old + other excluded


def test_digest_no_recent_content(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    _seed(repo, owner_id=1, text="old", date=_iso(30))
    api = FakeApi()
    probe = _Probe()
    probe.services = SimpleNamespace(repository=repo, api=api,
                                     settings=SimpleNamespace(gemini_api_key=None))
    probe._handle_digest(10, 1, "7")
    assert "No new content" in api.messages[-1][1]


def test_digest_fallback_without_api_key(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    _seed(repo, owner_id=1, text="recent one", date=_iso(2), title="AI News")
    api = FakeApi()
    probe = _Probe()
    probe.services = SimpleNamespace(repository=repo, api=api,
                                     settings=SimpleNamespace(gemini_api_key=None))
    probe._handle_digest(10, 1, "")
    msg = api.messages[-1][1]
    assert "Digest" in msg and "1 new item(s)" in msg and "AI News" in msg


def test_digest_uses_summarize_with_api_key(tmp_path, monkeypatch):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    _seed(repo, owner_id=1, text="recent content", date=_iso(1))
    api = FakeApi()
    probe = _Probe()
    probe.services = SimpleNamespace(
        repository=repo, api=api,
        settings=SimpleNamespace(gemini_api_key="key", vertex_project_id=None, vertex_region="us-central1"),
    )
    captured = {}

    class FakeSearch:
        def summarize(self, *, scope_label, items, **kwargs):
            captured["scope"] = scope_label
            captured["n"] = len(items)
            return "AI digest body"

    monkeypatch.setattr(probe, "_search_service_for_user", lambda user: FakeSearch())
    monkeypatch.setattr(probe, "_vertex_search_config", lambda user: None)

    probe._handle_digest(10, 1, "3")
    assert "the last 3 day(s)" in captured["scope"] and captured["n"] == 1
    assert "AI digest body" in api.messages[-1][1]
