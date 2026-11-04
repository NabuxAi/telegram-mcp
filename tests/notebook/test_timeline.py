from types import SimpleNamespace

from telegram_notebook.db import Repository
from telegram_notebook.timeline import build_timeline, period_key


def test_period_key_buckets_and_rejects_bad_dates():
    assert period_key("2026-06-11T10:30:00+00:00", "month") == "2026-06"
    assert period_key("2026-06-11T10:30:00+00:00", "day") == "2026-06-11"
    assert period_key("", "month") is None
    assert period_key("not-a-date", "month") is None
    assert period_key("2026/06/11", "month") is None  # wrong separator


def test_build_timeline_groups_newest_first_by_month():
    items = [
        {"message_date": "2026-06-01T00:00:00", "text": "june one", "channel_title": "A"},
        {"message_date": "2026-06-20T00:00:00", "text": "june two", "channel_url": "https://t.me/b"},
        {"message_date": "2026-04-15T00:00:00", "text": "april", "channel_title": "A"},
        {"message_date": "bad", "text": "ignored"},
    ]
    periods = build_timeline(items, granularity="month")
    assert [p["period"] for p in periods] == ["2026-06", "2026-04"]  # newest first, bad date dropped
    june = periods[0]
    assert june["count"] == 2
    assert june["sources"] == ["A", "https://t.me/b"]
    assert june["sample"] == "june one"  # first non-empty text in the bucket


def test_build_timeline_day_granularity():
    items = [
        {"message_date": "2026-06-11T09:00:00", "text": "a"},
        {"message_date": "2026-06-11T18:00:00", "text": "b"},
        {"message_date": "2026-06-12T01:00:00", "text": "c"},
    ]
    periods = build_timeline(items, granularity="day")
    assert {p["period"]: p["count"] for p in periods} == {"2026-06-11": 2, "2026-06-12": 1}


def _seed(repo: Repository, *, owner_id: int, channel_url: str, title: str, date: str, text: str, tag: str | None = None):
    cid = repo.upsert_channel(owner_id=owner_id, telegram_id=hash(channel_url) % 10_000, channel_url=channel_url, title=title, username=None)
    mid = repo.create_or_get_message(channel_id=cid, telegram_message_id=hash(text) % 10_000, message_date=date, message_url="u", caption=None)
    media_id = repo.create_or_get_media(message_id=mid, file_name="t", file_path="", mime_type="text/plain", media_kind="text", duration_seconds=None, file_size_bytes=1)
    repo.mark_media_transcribed(media_item_id=media_id, transcript_text=text)
    if tag:
        repo.tag_media(owner_id=owner_id, media_item_id=media_id, tags={tag})
    return media_id


def test_timeline_items_scoped_and_ordered(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    _seed(repo, owner_id=1, channel_url="https://t.me/a", title="A", date="2026-06-01T00:00:00", text="newer", tag="news")
    _seed(repo, owner_id=1, channel_url="https://t.me/a", title="A", date="2026-01-01T00:00:00", text="older")
    _seed(repo, owner_id=2, channel_url="https://t.me/a", title="A", date="2026-06-01T00:00:00", text="other user")

    rows = repo.timeline_items(owner_id=1)
    assert [r["text"] for r in rows] == ["newer", "older"]  # newest first
    assert all("other user" != r["text"] for r in rows)  # per-user isolation

    # tag scoping
    tagged = repo.timeline_items(owner_id=1, tag="news")
    assert [r["text"] for r in tagged] == ["newer"]


def test_mcp_timeline_tool(tmp_path):
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

    def call(args):
        return server.handle_request({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "timeline", "arguments": args},
        })["result"]["content"][0]["text"]

    listed = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert "timeline" in {t["name"] for t in listed["result"]["tools"]}
    assert "No dated content" in call({})

    _seed(repo, owner_id=1, channel_url="https://t.me/a", title="A", date="2026-06-01T00:00:00", text="hello")
    assert "2026-06: 1" in call({})
