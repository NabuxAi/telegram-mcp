import asyncio
from types import SimpleNamespace

from telegram_notebook import toolset


class FakeMessage:
    def __init__(self, id, message, date=None, sender=None):
        self.id = id
        self.message = message
        self.date = date
        self._sender = sender

    async def get_sender(self):
        return self._sender


class FakeClient:
    def __init__(self, *, me=None, scheduled=None, messages=None, entity=None):
        self._me = me
        self._scheduled = scheduled or []
        self._messages = messages or []
        self._entity = entity
        self.sent = []
        self.calls = []

    async def get_me(self):
        return self._me

    async def get_messages(self, peer, scheduled=False, limit=100):
        return self._scheduled if scheduled else self._messages

    async def get_entity(self, peer):
        return self._entity

    async def get_input_entity(self, peer):
        return self._entity

    def iter_messages(self, entity, limit=200):
        messages = self._messages

        async def _gen():
            for m in messages:
                yield m

        return _gen()

    async def send_message(self, target, text):
        self.sent.append((target, text))
        return FakeMessage(id=999, message=text)

    async def __call__(self, request):
        self.calls.append(request)
        return None


# --- pure helpers ---

def test_display_name_variants():
    assert toolset._display_name(SimpleNamespace(first_name="Ada", last_name="Lovelace")) == "Ada Lovelace"
    assert toolset._display_name(SimpleNamespace(title="My Channel")) == "My Channel"
    assert toolset._display_name(SimpleNamespace(username="handle")) == "handle"
    assert toolset._display_name(None) == "Unknown"


def test_format_account():
    info = {"first_name": "Ada", "last_name": None, "username": "ada", "id": 7, "phone": "123", "premium": True, "verified": False}
    out = toolset.format_account(info)
    assert "Ada" in out and "@ada" in out and "7" in out and "123" in out and "✅" in out


def test_format_scheduled_empty_and_items():
    assert "No scheduled messages" in toolset.format_scheduled("@chan", [])
    out = toolset.format_scheduled("@chan", [{"id": 3, "date": "2026-06-17", "text": "hi there"}])
    assert "@chan" in out and "#3" in out and "hi there" in out


def test_build_llm_export_oldest_first():
    rows = [  # newest-first as Telethon returns
        {"id": 2, "date": "2026-06-02", "sender": "Bob", "text": "second"},
        {"id": 1, "date": "2026-06-01", "sender": "Ada", "text": "first"},
    ]
    md = toolset.build_llm_export("My Chat", rows)
    assert md.startswith("# Telegram chat — My Chat")
    assert "2 message(s)" in md
    assert md.index("first") < md.index("second")  # emitted oldest-first
    assert "**Ada**" in md and "**Bob**" in md
    assert md.endswith("\n")


# --- async wrappers ---

def test_account_info():
    me = SimpleNamespace(id=5, first_name="Ada", last_name="L", username="ada", phone="1", premium=True, verified=False, bot=False)
    info = asyncio.run(toolset.account_info(FakeClient(me=me)))
    assert info["id"] == 5 and info["username"] == "ada" and info["premium"] is True


def test_list_scheduled_maps_fields():
    client = FakeClient(scheduled=[FakeMessage(1, "hello", date="d1"), FakeMessage(2, "", date="d2")])
    items = asyncio.run(toolset.list_scheduled(client, "@chan"))
    assert [it["id"] for it in items] == [1, 2]
    assert items[0]["text"] == "hello" and items[1]["text"] == ""


def test_resend_text_sends_and_returns_id():
    client = FakeClient()
    msg_id = asyncio.run(toolset.resend_text(client, "@user", "yo"))
    assert msg_id == 999 and client.sent == [("@user", "yo")]


def test_fetch_chat_messages_skips_empty_and_labels():
    entity = SimpleNamespace(title="My Chat")
    msgs = [FakeMessage(2, "world", date="d2", sender=SimpleNamespace(first_name="Bob")),
            FakeMessage(3, "", date="d3", sender=None),  # skipped (no text)
            FakeMessage(1, "hello", date="d1", sender=SimpleNamespace(first_name="Ada"))]
    label, rows = asyncio.run(toolset.fetch_chat_messages(FakeClient(messages=msgs, entity=entity), "@chan"))
    assert label == "My Chat"
    assert [r["id"] for r in rows] == [2, 1]  # empty one dropped
    assert rows[0]["sender"] == "Bob"


def test_cancel_scheduled_invokes_request():
    client = FakeClient(entity=SimpleNamespace(id=1))
    n = asyncio.run(toolset.cancel_scheduled(client, "@chan", [4, 5]))
    assert n == 2 and len(client.calls) == 1
