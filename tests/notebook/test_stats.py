from types import SimpleNamespace

from telegram_notebook.db import Repository
from telegram_notebook.stats import format_stats


def test_format_stats_empty():
    assert "empty" in format_stats({"items": 0})


def test_format_stats_full():
    out = format_stats({
        "items": 5, "sources": 2, "tags": 3,
        "by_kind": {"text": 4, "voice": 1},
        "first_date": "2026-01-02T00:00:00", "last_date": "2026-06-10T00:00:00",
    })
    assert "Items: 5" in out
    assert "Sources: 2" in out and "Tags: 3" in out
    assert "text: 4" in out and "voice: 1" in out
    assert "2026-01-02 → 2026-06-10" in out


def _seed(repo, *, owner_id, channel_url, title, kind, text, date="2026-06-01T00:00:00", tag=None):
    cid = repo.upsert_channel(owner_id=owner_id, telegram_id=hash(channel_url) % 9999, channel_url=channel_url, title=title, username=None)
    mid = repo.create_or_get_message(channel_id=cid, telegram_message_id=hash(text) % 9999, message_date=date, message_url="u", caption=None)
    media_id = repo.create_or_get_media(message_id=mid, file_name="t", file_path="", mime_type="text/plain", media_kind=kind, duration_seconds=None, file_size_bytes=1)
    repo.mark_media_transcribed(media_item_id=media_id, transcript_text=text)
    if tag:
        repo.tag_media(owner_id=owner_id, media_item_id=media_id, tags={tag})


def test_archive_stats_aggregates_and_scopes(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    _seed(repo, owner_id=1, channel_url="https://t.me/a", title="A", kind="text", text="one", date="2026-01-01T00:00:00", tag="news")
    _seed(repo, owner_id=1, channel_url="https://t.me/a", title="A", kind="voice", text="two", date="2026-03-01T00:00:00")
    _seed(repo, owner_id=1, channel_url="https://t.me/b", title="B", kind="text", text="three", date="2026-05-01T00:00:00")
    _seed(repo, owner_id=2, channel_url="https://t.me/a", title="A", kind="text", text="other user", date="2026-02-01T00:00:00")

    s = repo.archive_stats(owner_id=1)
    assert s["items"] == 3
    assert s["sources"] == 2
    assert s["tags"] == 1
    assert s["by_kind"] == {"text": 2, "voice": 1}
    assert s["first_date"][:10] == "2026-01-01" and s["last_date"][:10] == "2026-05-01"

    # other user isolated
    assert repo.archive_stats(owner_id=2)["items"] == 1
    # empty owner
    assert repo.archive_stats(owner_id=99)["items"] == 0


def test_mcp_archive_stats_tool(tmp_path):
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
    assert "archive_stats" in {t["name"] for t in listed["result"]["tools"]}

    _seed(repo, owner_id=1, channel_url="https://t.me/a", title="A", kind="text", text="hi")
    out = server.handle_request({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "archive_stats", "arguments": {}},
    })["result"]["content"][0]["text"]
    assert "Items: 1" in out
