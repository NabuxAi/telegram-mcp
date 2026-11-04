from types import SimpleNamespace

from telegram_notebook.bot import NotebookBot
from telegram_notebook.db import Repository
from telegram_notebook.recent import recent_rows


class _Probe(NotebookBot):
    def __init__(self):
        pass


class FakeApi:
    def __init__(self):
        self.messages = []

    def send_message(self, chat_id=None, text=None, **kwargs):
        self.messages.append((chat_id, text))


def test_recent_rows_limits_and_normalizes():
    items = [
        {"message_date": "2026-06-10T09:00:00", "text": "  multi   space\ntext  ", "channel_title": "A", "message_url": "u1"},
        {"message_date": "2026-06-09T09:00:00", "text": "x" * 500, "channel_url": "https://t.me/b"},
        {"message_date": "2026-06-08T09:00:00", "text": "third", "channel_title": "C"},
    ]
    rows = recent_rows(items, limit=2, snippet_chars=100)
    assert len(rows) == 2
    assert rows[0] == {"source": "A", "date": "2026-06-10", "snippet": "multi space text", "url": "u1"}
    assert rows[1]["source"] == "https://t.me/b"
    assert len(rows[1]["snippet"]) == 100  # truncated


def test_recent_rows_unknown_source():
    rows = recent_rows([{"text": "hi"}])
    assert rows[0]["source"] == "Unknown source" and rows[0]["date"] == ""


def _seed(repo, *, owner_id, text, date):
    cid = repo.upsert_channel(owner_id=owner_id, telegram_id=1, channel_url="https://t.me/c", title="C", username=None)
    mid = repo.create_or_get_message(channel_id=cid, telegram_message_id=hash(text) % 9999, message_date=date, message_url="https://t.me/c/1", caption=None)
    media_id = repo.create_or_get_media(message_id=mid, file_name="t", file_path="", mime_type="text/plain", media_kind="text", duration_seconds=None, file_size_bytes=1)
    repo.mark_media_transcribed(media_item_id=media_id, transcript_text=text)


def test_handle_recent(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    _seed(repo, owner_id=1, text="newest", date="2026-06-10T00:00:00")
    _seed(repo, owner_id=1, text="older", date="2026-06-01T00:00:00")
    probe = _Probe()
    api = FakeApi()
    probe.services = SimpleNamespace(repository=repo, api=api)

    probe._handle_recent(10, 1, "")
    msg = api.messages[-1][1]
    assert "Recent (2)" in msg
    assert msg.index("newest") < msg.index("older")  # newest first


def test_handle_recent_empty(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    probe = _Probe()
    api = FakeApi()
    probe.services = SimpleNamespace(repository=repo, api=api)
    probe._handle_recent(10, 1, "")
    assert "No content yet" in api.messages[-1][1]


def test_mcp_list_recent_tool(tmp_path):
    from telegram_notebook.embeddings import EmbeddingService
    from telegram_notebook.mcp_server import McpServer
    from telegram_notebook.search import SearchService

    repo = Repository(tmp_path / "store.db")
    repo.init()
    server = McpServer(
        repository=repo,
        search_service=SearchService(repo, EmbeddingService(provider="openai", api_key=None, model="x")),
        settings=SimpleNamespace(gemini_api_key=None, vertex_project_id=None, vertex_region="us-central1"),
        owner_id=1,
    )
    listed = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert "list_recent" in {t["name"] for t in listed["result"]["tools"]}

    _seed(repo, owner_id=1, text="hello recent", date="2026-06-10T00:00:00")
    out = server.handle_request({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "list_recent", "arguments": {"limit": 5}},
    })["result"]["content"][0]["text"]
    assert "hello recent" in out
