from types import SimpleNamespace

from telegram_notebook.bot import NotebookBot
from telegram_notebook.db import Repository


class FakeApi:
    def __init__(self):
        self.sent: list[tuple] = []

    def send_message(self, chat_id=None, text=None, **kwargs):
        self.sent.append((chat_id, text))
        return {"ok": True}


class RaisingApi:
    def send_message(self, *args, **kwargs):
        raise RuntimeError("send failed")


def _bot(repo, api):
    # _auto_forward / _handle_setarchive only touch self.services, so a lightweight
    # stand-in avoids NotebookBot's heavy __init__ (event loop, worker, executor).
    return SimpleNamespace(services=SimpleNamespace(api=api, repository=repo))


# --- _auto_forward (pure decision + formatting) ---

def test_auto_forward_sends_when_archive_and_tags():
    api = FakeApi()
    ok = NotebookBot._auto_forward(
        _bot(None, api),
        user={"archive_chat_id": "@arch", "bot_user_id": 1},
        label="AI News",
        tags={"AI", "Tools"},
        text="a note about claude",
        message_url="https://t.me/ainews/42",
    )
    assert ok is True
    assert len(api.sent) == 1
    chat, text = api.sent[0]
    assert chat == "@arch"
    assert "AI News" in text
    assert "#AI" in text and "#Tools" in text  # tags rendered
    assert "a note about claude" in text
    assert "https://t.me/ainews/42" in text


def test_auto_forward_escapes_html_in_user_fields():
    # parse_mode is HTML, so <, > and & in user-controlled fields must be escaped
    # or Telegram rejects the message and the archive send silently fails.
    api = FakeApi()
    ok = NotebookBot._auto_forward(
        _bot(None, api),
        user={"archive_chat_id": "@arch", "bot_user_id": 1},
        label="A & B <News>",
        tags={"R&D"},
        text="price < 5 & rising > before",
        message_url="https://t.me/c/1?a=1&b=2",
    )
    assert ok is True
    text = api.sent[0][1]
    assert "&amp;" in text and "&lt;" in text and "&gt;" in text
    # no raw unescaped special chars from user fields leak into the markup
    assert "A & B" not in text and "<News>" not in text
    assert "price < 5" not in text
    # our own intentional <b> wrapper is still present
    assert "<b>" in text and "</b>" in text


def test_auto_forward_skips_without_archive_or_tags():
    api = FakeApi()
    # no tags matched
    assert NotebookBot._auto_forward(
        _bot(None, api), user={"archive_chat_id": "@a", "bot_user_id": 1},
        label="l", tags=set(), text="t", message_url=None,
    ) is False
    # no archive configured
    assert NotebookBot._auto_forward(
        _bot(None, api), user={"bot_user_id": 1},
        label="l", tags={"AI"}, text="t", message_url=None,
    ) is False
    # no user at all
    assert NotebookBot._auto_forward(
        _bot(None, api), user=None, label="l", tags={"AI"}, text="t", message_url=None,
    ) is False
    assert api.sent == []


def test_auto_forward_swallows_send_error():
    ok = NotebookBot._auto_forward(
        _bot(None, RaisingApi()),
        user={"archive_chat_id": "@a", "bot_user_id": 1},
        label="l", tags={"t"}, text="x", message_url=None,
    )
    assert ok is False  # send failure does not raise


# --- /setarchive command + repository persistence ---

def test_setarchive_set_show_and_clear(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    repo.upsert_bot_user(bot_user_id=1, chat_id=10, username=None, first_name=None)
    api = FakeApi()
    fake = _bot(repo, api)

    # fresh user: no archive yet -> usage hint
    NotebookBot._handle_setarchive(fake, 10, 1, "")
    assert "Usage" in api.sent[-1][1]

    # set it
    NotebookBot._handle_setarchive(fake, 10, 1, "@archive")
    assert repo.get_bot_user(bot_user_id=1)["archive_chat_id"] == "@archive"

    # show current
    NotebookBot._handle_setarchive(fake, 10, 1, "")
    assert "@archive" in api.sent[-1][1]

    # disable
    NotebookBot._handle_setarchive(fake, 10, 1, "off")
    assert repo.get_bot_user(bot_user_id=1)["archive_chat_id"] is None


def test_archive_column_migrated_and_scoped(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    repo.upsert_bot_user(bot_user_id=1, chat_id=10, username=None, first_name=None)
    repo.upsert_bot_user(bot_user_id=2, chat_id=20, username=None, first_name=None)

    # migration added the column (default NULL)
    assert repo.get_bot_user(bot_user_id=1)["archive_chat_id"] is None

    repo.set_archive_chat(bot_user_id=1, archive_chat_id="@only_user_one")
    assert repo.get_bot_user(bot_user_id=1)["archive_chat_id"] == "@only_user_one"
    assert repo.get_bot_user(bot_user_id=2)["archive_chat_id"] is None  # per-user
