import asyncio
from pathlib import Path
from types import SimpleNamespace

from telegram_notebook.bot import NotebookBot
from telegram_notebook.db import Repository


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

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

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


class _Probe(NotebookBot):
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.watcher = SimpleNamespace(start_for_user=lambda u: None, stop_for_user=lambda i: None)


class FakeApi:
    def __init__(self):
        self.messages = []
        self.documents = []

    def send_message(self, chat_id=None, text=None, **kwargs):
        self.messages.append((chat_id, text))

    def send_document(self, *, chat_id, document_path, caption=None):
        self.documents.append((chat_id, Path(document_path).read_text(encoding="utf-8"), caption))


def _connected_repo(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    repo.upsert_bot_user(bot_user_id=1, chat_id=10, username="u", first_name="U")
    repo.save_bot_user_session(bot_user_id=1, phone="+1", api_id=7, api_hash="h", session_string="sess", connected_at="now")
    return repo


def _probe(repo, api, fake_client=None):
    probe = _Probe()
    probe.services = SimpleNamespace(repository=repo, api=api, settings=SimpleNamespace())
    if fake_client is not None:
        probe._user_client = lambda user: fake_client
    return probe


# --- guards ---

def test_handlers_require_connect(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()  # no connected user
    api = FakeApi()
    probe = _probe(repo, api)
    for fn in (lambda: probe._handle_account(10, 1),
               lambda: probe._handle_scheduled(10, 1, "@c"),
               lambda: probe._handle_resend(10, 1, "@c hi")):
        api.messages.clear()
        fn()
        assert "Please /connect first." in api.messages[-1][1]


# --- account ---

def test_handle_account(tmp_path):
    repo = _connected_repo(tmp_path)
    api = FakeApi()
    me = SimpleNamespace(id=5, first_name="Ada", last_name=None, username="ada", phone="1", premium=False, verified=False, bot=False)
    probe = _probe(repo, api, FakeClient(me=me))
    probe._handle_account(10, 1)
    assert "Ada" in api.messages[-1][1] and "@ada" in api.messages[-1][1]


# --- scheduled (list + cancel) ---

def test_handle_scheduled_list(tmp_path):
    repo = _connected_repo(tmp_path)
    api = FakeApi()
    client = FakeClient(scheduled=[FakeMessage(1, "later", date="d1")])
    probe = _probe(repo, api, client)
    probe._handle_scheduled(10, 1, "@chan")
    assert "Scheduled" in api.messages[-1][1] and "#1" in api.messages[-1][1]


def test_handle_scheduled_cancel(tmp_path):
    repo = _connected_repo(tmp_path)
    api = FakeApi()
    client = FakeClient(entity=SimpleNamespace(id=1))
    probe = _probe(repo, api, client)
    probe._handle_scheduled(10, 1, "@chan cancel 1")
    assert len(client.calls) == 1
    assert "Cancelled scheduled message #1" in api.messages[-1][1]


# --- llm export ---

def test_handle_llm_export_sends_document(tmp_path):
    repo = _connected_repo(tmp_path)
    api = FakeApi()
    entity = SimpleNamespace(title="My Chat")
    msgs = [FakeMessage(1, "hello", date="d1", sender=SimpleNamespace(first_name="Ada"))]
    probe = _probe(repo, api, FakeClient(messages=msgs, entity=entity))
    probe._handle_llm_export(10, 1, "@chan")
    assert len(api.documents) == 1
    _, content, caption = api.documents[0]
    assert "hello" in content and content.startswith("# Telegram chat — My Chat")
    assert "LLM export" in caption


def test_handle_llm_export_empty(tmp_path):
    repo = _connected_repo(tmp_path)
    api = FakeApi()
    probe = _probe(repo, api, FakeClient(messages=[], entity=SimpleNamespace(title="Empty")))
    probe._handle_llm_export(10, 1, "@chan")
    assert api.documents == []
    assert "No text messages" in api.messages[-1][1]


# --- resend ---

def test_handle_resend(tmp_path):
    repo = _connected_repo(tmp_path)
    api = FakeApi()
    client = FakeClient()
    probe = _probe(repo, api, client)
    probe._handle_resend(10, 1, "@user hello world")
    assert client.sent == [("@user", "hello world")]
    assert "Sent to @user" in api.messages[-1][1]


def test_handle_resend_usage(tmp_path):
    repo = _connected_repo(tmp_path)
    api = FakeApi()
    probe = _probe(repo, api, FakeClient())
    probe._handle_resend(10, 1, "@user")  # missing text
    assert "Usage:" in api.messages[-1][1]


# --- watchdeleted + deleted ---

def test_handle_watchdeleted_toggle(tmp_path):
    repo = _connected_repo(tmp_path)
    api = FakeApi()
    probe = _probe(repo, api)
    probe._handle_watchdeleted(10, 1, "on")
    assert repo.get_bot_user(bot_user_id=1)["watch_deleted"] == 1
    assert "ON" in api.messages[-1][1]
    probe._handle_watchdeleted(10, 1, "off")
    assert repo.get_bot_user(bot_user_id=1)["watch_deleted"] == 0
    assert "OFF" in api.messages[-1][1]


def test_handle_deleted_lists_recovered(tmp_path):
    repo = _connected_repo(tmp_path)
    api = FakeApi()
    repo.add_recovered(owner_id=1, rows=[{"chat_id": 1, "message_id": 1, "chat_title": "Chat", "sender": "Ada", "text": "gone"}], deleted_at="t")
    probe = _probe(repo, api)
    probe._handle_deleted(10, 1, "")
    assert "Recovered deleted" in api.messages[-1][1] and "gone" in api.messages[-1][1]


def test_handle_deleted_empty(tmp_path):
    repo = _connected_repo(tmp_path)
    api = FakeApi()
    probe = _probe(repo, api)
    probe._handle_deleted(10, 1, "")
    assert "No recovered deleted messages" in api.messages[-1][1]
