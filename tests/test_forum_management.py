"""Unit tests for forum management. No Telegram network calls are made."""
import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest


class FakeForum:
    def __init__(self, *, forum=True, creator=False, manage_topics=False):
        self.id = 55
        self.forum = forum
        self.megagroup = True
        self.creator = creator
        self.admin_rights = SimpleNamespace(manage_topics=manage_topics)


class Topic:
    def __init__(self, tid, title="Existing", *, closed=False, hidden=False):
        self.id = tid
        self.title = title
        self.closed = closed
        self.hidden = hidden


class MCP:
    def __init__(self):
        self.registered = {}

    def tool(self, name, annotations=None):
        def decorate(fn):
            self.registered[name] = fn
            return fn
        return decorate


class Client:
    def __init__(self):
        self.topics = [Topic(77)]
        self.requests = []
        self.latest_kwargs = {}

    async def __call__(self, request):
        from app import tools
        self.requests.append(request)
        f = tools.functions
        if isinstance(request, f.messages.GetForumTopicsRequest):
            return SimpleNamespace(
                topics=self.topics, count=len(self.topics),
                messages=[SimpleNamespace(id=400, date=datetime(2026, 11, 1, tzinfo=timezone.utc))],
            )
        if isinstance(request, f.messages.GetForumTopicsByIDRequest):
            return SimpleNamespace(topics=[t for t in self.topics if t.id in request.topics])
        if isinstance(request, f.messages.CreateForumTopicRequest):
            self.topics.append(Topic(88, request.title))
            action = tools.types.MessageActionTopicCreate(title=request.title, icon_color=0x6FB9F0)
            return SimpleNamespace(updates=[SimpleNamespace(
                message=SimpleNamespace(id=88, action=action))])
        if isinstance(request, f.messages.EditForumTopicRequest):
            t = next(t for t in self.topics if t.id == request.topic_id)
            if request.title is not None:
                t.title = request.title
            if request.closed is not None:
                t.closed = request.closed
            if request.hidden is not None:
                t.hidden = request.hidden
            return SimpleNamespace(updates=[])
        if isinstance(request, f.messages.UpdatePinnedForumTopicRequest):
            return SimpleNamespace(updates=[])
        if isinstance(request, f.messages.DeleteTopicHistoryRequest):
            self.topics = [t for t in self.topics if t.id != request.top_msg_id]
            return SimpleNamespace(updates=[])
        if isinstance(request, f.channels.ToggleForumRequest):
            return SimpleNamespace(updates=[])
        raise AssertionError(f"Unexpected request: {request!r}")

    async def iter_messages(self, _ent, **kwargs):
        self.latest_kwargs = kwargs
        yield SimpleNamespace(id=10)


@pytest.fixture
def forum(monkeypatch):
    from app import tools
    old_mcp, old_pool = tools._mcp, tools._pool
    mcp, channel, client = MCP(), FakeForum(creator=True), Client()
    tools.register(mcp, SimpleNamespace())

    async def entry(*_args, **_kwargs):
        return SimpleNamespace(client=client)

    async def resolve(*_args, **_kwargs):
        return channel

    monkeypatch.setattr(tools, "entry", entry)
    monkeypatch.setattr(tools, "resolve", resolve)
    monkeypatch.setattr(tools, "guarded", lambda _entry, fn: fn())
    monkeypatch.setattr(tools, "chat_id_of", lambda _ent: -10055)
    monkeypatch.setattr(tools, "ser_message", lambda m, *_args: {"id": m.id})
    monkeypatch.setattr(tools.types, "Channel", FakeForum)
    monkeypatch.setattr(tools.types, "ForumTopic", Topic)
    def run(name, **kwargs):
        return json.loads(asyncio.run(mcp.registered[name](chat="@example", **kwargs)))
    try:
        yield run, channel, client
    finally:
        tools._mcp, tools._pool = old_mcp, old_pool


def test_create_requires_manage_rights_and_forum(forum):
    run, channel, client = forum
    channel.creator = False
    assert not run("telegram_create_forum_topic", title="Test")["ok"]
    assert not client.requests
    channel.admin_rights.manage_topics = True
    channel.forum = False
    assert not run("telegram_create_forum_topic", title="Test")["ok"]
    assert not client.requests


def test_create_returns_root_id_and_rejects_duplicate(forum):
    run, _, client = forum
    res = run("telegram_create_forum_topic", title="   AI lessons   ")
    assert res["ok"] and res["topic_id"] == 88 and res["title"] == "AI lessons"
    assert len(client.topics) == 2
    again = run("telegram_create_forum_topic", title="ai LESSONS")
    assert not again["ok"] and "already exists" in again["error"]
    assert len(client.topics) == 2
    assert not run("telegram_create_forum_topic", title="   ")["ok"]


def test_edit_close_and_reopen_forum_topic(forum):
    run, _, client = forum
    r = run("telegram_edit_forum_topic", topic_id=77, title="Updated", closed=True)
    assert r["ok"] and r["changes"] == {"title": "Updated", "closed": True}
    assert client.topics[0].closed is True
    assert run("telegram_edit_forum_topic", topic_id=77, closed=False)["ok"]
    assert client.topics[0].closed is False
    assert not run("telegram_edit_forum_topic", topic_id=77)["ok"]
    assert not run("telegram_edit_forum_topic", topic_id=77, hidden=True)["ok"]


def test_pin_and_delete_require_explicit_confirmation(forum):
    run, _, client = forum
    assert run("telegram_pin_forum_topic", topic_id=77, pinned=True)["ok"]
    before = len(client.requests)
    assert not run("telegram_delete_forum_topic", topic_id=77)["ok"]
    assert len(client.requests) == before
    assert not run("telegram_delete_forum_topic", topic_id=1, confirm_delete=True)["ok"]
    assert run("telegram_delete_forum_topic", topic_id=77, confirm_delete=True)["ok"]
    assert client.topics == []


def test_toggle_requires_creator_and_confirm(forum):
    run, channel, client = forum
    assert not run("telegram_set_forum_enabled", enabled=False)["ok"]
    assert not client.requests
    channel.creator = False
    channel.admin_rights.manage_topics = True
    assert not run("telegram_set_forum_enabled", enabled=True, confirm_change=True)["ok"]
    channel.creator = True
    assert run("telegram_set_forum_enabled", enabled=True, confirm_change=True)["ok"]


def test_topic_scoped_read_uses_reply_root(forum):
    run, _, client = forum
    res = run("telegram_get_messages", topic_id=77, limit=15)
    assert res["ok"] and res["topic_id"] == 77
    assert client.latest_kwargs["reply_to"] == 77
    assert client.latest_kwargs["limit"] == 15
    assert not run("telegram_get_messages", topic_id=77, reply_to=10)["ok"]
    assert not run("telegram_get_messages", topic_id=999)["ok"]
