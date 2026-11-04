"""Offline forum topic tests: never contact Telegram or send a real message."""
import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest


class FakeChannel:
    def __init__(self, forum=True):
        self.forum = forum
        self.id = 123


class FakeTopic:
    def __init__(self, tid, title="Daily prompts", *, closed=False, hidden=False, top_message=400):
        self.id = tid
        self.title = title
        self.closed = closed
        self.hidden = hidden
        self.pinned = False
        self.top_message = top_message
        self.unread_count = 2


class FakeMCP:
    def __init__(self):
        self.registered = {}

    def tool(self, name, annotations=None):
        def deco(func):
            self.registered[name] = func
            return func
        return deco


class FakeClient:
    def __init__(self, topics=None):
        self.topics = topics if topics is not None else [FakeTopic(77)]
        self.requests = []
        self.sends = []
        self.files = []

    async def __call__(self, request):
        from app import tools
        self.requests.append(request)
        if isinstance(request, tools.functions.messages.GetForumTopicsRequest):
            return SimpleNamespace(
                topics=self.topics, count=len(self.topics),
                messages=[SimpleNamespace(id=400, date=datetime(2026, 11, 1, tzinfo=timezone.utc))],
            )
        if isinstance(request, tools.functions.messages.GetForumTopicsByIDRequest):
            matches = [t for t in self.topics if t.id in request.topics]
            return SimpleNamespace(topics=matches)
        raise AssertionError(f"Unexpected Telegram request: {request!r}")

    async def send_message(self, *args, **kwargs):
        self.sends.append((args, kwargs))
        return SimpleNamespace(id=1001)

    async def send_file(self, *args, **kwargs):
        self.files.append((args, kwargs))
        return SimpleNamespace(id=1002)


@pytest.fixture
def setup_forum(monkeypatch):
    from app import tools
    previous_mcp, previous_pool = tools._mcp, tools._pool
    mcp = FakeMCP()
    tools.register(mcp, SimpleNamespace())
    channel = FakeChannel()
    client = FakeClient()
    entry = SimpleNamespace(client=client)

    async def get_entry(*args, **kwargs):
        return entry

    async def resolve(_entry, _chat):
        return channel

    monkeypatch.setattr(tools.types, "Channel", FakeChannel)
    monkeypatch.setattr(tools.types, "ForumTopic", FakeTopic)
    monkeypatch.setattr(tools, "entry", get_entry)
    monkeypatch.setattr(tools, "resolve", resolve)
    monkeypatch.setattr(tools, "guarded", lambda _entry, fn: fn())
    monkeypatch.setattr(tools, "chat_id_of", lambda ent: -100123)
    monkeypatch.setattr(tools, "ser_message", lambda msg: {"id": msg.id})
    def synchronous(fn):
        def call(**kwargs):
            return asyncio.run(fn(**kwargs))
        return call

    try:
        yield {name: synchronous(fn) for name, fn in mcp.registered.items()}, channel, client
    finally:
        tools._mcp, tools._pool = previous_mcp, previous_pool


def test_topic_list_metadata_only_and_cursor(setup_forum):
    registered, _, client = setup_forum
    result = json.loads(registered["telegram_list_forum_topics"](
        chat="@forum", limit=20, query="Daily"))
    assert result["ok"] and result["topics"] == [{
        "id": 77, "title": "Daily prompts", "closed": False,
        "hidden": False, "pinned": False, "unread_count": 2,
    }]
    assert result["next_offset"] == {
        "offset_date": 1793491200, "offset_id": 400, "offset_topic": 77,
    }
    request = client.requests[0]
    assert request.q == "Daily" and request.limit == 20
    assert not client.sends and not client.files


def test_forum_required_for_list_and_send(setup_forum):
    registered, channel, client = setup_forum
    channel.forum = False
    for name, args in [
        ("telegram_list_forum_topics", {"chat": "@plain"}),
        ("telegram_send_message", {"chat": "@plain", "text": "Hello", "topic_id": 77}),
    ]:
        result = json.loads(registered[name](**args))
        assert not result["ok"] and "forum-enabled" in result["error"]
    assert not client.requests and not client.sends


def test_valid_topic_routes_text_to_topic_root(setup_forum):
    registered, _, client = setup_forum
    result = json.loads(registered["telegram_send_message"](
        chat="@forum", text="An example", topic_id=77))
    assert result["ok"] and result["topic_id"] == 77
    assert result["topic_title"] == "Daily prompts"
    assert client.sends[0][1]["reply_to"] == 77
    assert len(client.requests) == 1


def test_general_topic_uses_main_timeline(setup_forum):
    registered, _, client = setup_forum
    client.topics = [FakeTopic(1, "General", top_message=11)]
    result = json.loads(registered["telegram_send_message"](
        chat="@forum", text="General post", topic_id=1))
    assert result["ok"] and result["topic_id"] == 1
    assert client.sends[0][1]["reply_to"] is None


def test_file_post_targets_selected_topic_without_downloading(setup_forum):
    registered, _, client = setup_forum
    result = json.loads(registered["telegram_send_message"](
        chat="@forum", text="Photo", file_base64="aGVsbG8=", file_name="photo.jpg", topic_id=77))
    assert result["ok"]
    assert client.files[0][1]["reply_to"] == 77


@pytest.mark.parametrize("topics,tid", [
    ([FakeTopic(77)], 55),  # Unknown topic
    ([FakeTopic(77, closed=True)], 77),
    ([FakeTopic(77, hidden=True)], 77),
    ([], 77),  # Deleted topic
])
def test_rejects_missing_closed_hidden_deleted_topics_before_send(setup_forum, topics, tid):
    registered, _, client = setup_forum
    client.topics = topics
    result = json.loads(registered["telegram_send_message"](
        chat="@forum", text="Must not post", file_url="https://example.org/photo.jpg", topic_id=tid))
    assert not result["ok"]
    assert not client.sends and not client.files


@pytest.mark.parametrize("topic_id,reply_to", [(77, 123), (0, None), (-5, None)])
def test_rejects_ambiguous_or_invalid_topic_ids(setup_forum, topic_id, reply_to):
    registered, _, client = setup_forum
    result = json.loads(registered["telegram_send_message"](
        chat="@forum", text="Must not post", topic_id=topic_id, reply_to=reply_to))
    assert not result["ok"]
    assert not client.requests and not client.sends


def test_original_non_topic_send_behavior_unchanged(setup_forum):
    registered, _, client = setup_forum
    result = json.loads(registered["telegram_send_message"](
        chat="@plain", text="Legacy", reply_to=12))
    assert result["ok"] and result["topic_id"] is None
    assert client.sends[0][1]["reply_to"] == 12
    assert not client.requests
