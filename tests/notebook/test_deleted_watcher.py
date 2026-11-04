import threading
import time
from types import SimpleNamespace

from telegram_notebook.db import Repository
from telegram_notebook.deleted_watcher import DeletedWatcherManager, recover_deleted


def _repo(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    return repo


def _cache(repo, owner_id, chat_id, message_id, text):
    repo.cache_message(owner_id=owner_id, chat_id=chat_id, message_id=message_id,
                       chat_title="Chat", sender="Ada", text=text, message_date="2026-06-17")


# --- repository cache / recover ---

def test_cache_and_take(tmp_path):
    repo = _repo(tmp_path)
    _cache(repo, 1, 100, 10, "hello")
    _cache(repo, 1, 100, 11, "world")
    taken = repo.take_cached(owner_id=1, message_ids=[10])
    assert len(taken) == 1 and taken[0]["text"] == "hello"
    # taken rows are removed; the other stays
    assert repo.take_cached(owner_id=1, message_ids=[10]) == []
    assert len(repo.take_cached(owner_id=1, message_ids=[11])) == 1


def test_take_respects_owner_and_chat(tmp_path):
    repo = _repo(tmp_path)
    _cache(repo, 1, 100, 10, "mine")
    _cache(repo, 2, 100, 10, "theirs")
    assert repo.take_cached(owner_id=1, message_ids=[10], chat_id=999) == []  # wrong chat
    mine = repo.take_cached(owner_id=1, message_ids=[10], chat_id=100)
    assert len(mine) == 1 and mine[0]["text"] == "mine"


def test_recover_deleted_moves_to_recovered(tmp_path):
    repo = _repo(tmp_path)
    _cache(repo, 1, 100, 10, "secret")
    _cache(repo, 1, 100, 11, "kept")
    rows = recover_deleted(repo, owner_id=1, message_ids=[10], chat_id=100)
    assert len(rows) == 1 and rows[0]["text"] == "secret"
    recovered = repo.list_recovered(owner_id=1)
    assert len(recovered) == 1 and recovered[0]["text"] == "secret"
    # the non-deleted message is still cached (recoverable later)
    assert len(repo.take_cached(owner_id=1, message_ids=[11])) == 1


def test_recover_deleted_no_cache_is_noop(tmp_path):
    repo = _repo(tmp_path)
    assert recover_deleted(repo, owner_id=1, message_ids=[42], chat_id=1) == []
    assert repo.list_recovered(owner_id=1) == []


def test_list_recovered_newest_first_and_scoped(tmp_path):
    repo = _repo(tmp_path)
    repo.add_recovered(owner_id=1, rows=[{"chat_id": 1, "message_id": 1, "text": "a"}], deleted_at="t1")
    repo.add_recovered(owner_id=1, rows=[{"chat_id": 1, "message_id": 2, "text": "b"}], deleted_at="t2")
    repo.add_recovered(owner_id=2, rows=[{"chat_id": 1, "message_id": 3, "text": "other"}], deleted_at="t3")
    rows = repo.list_recovered(owner_id=1)
    assert [r["text"] for r in rows] == ["b", "a"]  # newest first
    assert repo.list_recovered(owner_id=2)[0]["text"] == "other"


# --- watch_deleted flag + owners ---

def test_watch_deleted_flag_and_owner_listing(tmp_path):
    repo = _repo(tmp_path)
    repo.upsert_bot_user(bot_user_id=1, chat_id=10, username="u", first_name="U")
    repo.save_bot_user_session(bot_user_id=1, phone="+1", api_id=7, api_hash="h", session_string="sess", connected_at="now")
    assert repo.list_watch_deleted_owners() == []  # not opted in yet
    repo.set_watch_deleted(bot_user_id=1, enabled=True)
    owners = repo.list_watch_deleted_owners()
    assert len(owners) == 1 and owners[0]["bot_user_id"] == 1 and owners[0]["session_string"] == "sess"
    repo.set_watch_deleted(bot_user_id=1, enabled=False)
    assert repo.list_watch_deleted_owners() == []


# --- manager lifecycle (with a fake client) ---

class FakeWatcherClient:
    def __init__(self):
        self._ev = threading.Event()
        self.started = threading.Event()
        self.disconnected = False

    def on(self, event):
        def deco(fn):
            return fn
        return deco

    def start(self):
        pass

    def run_until_disconnected(self):
        self.started.set()
        self._ev.wait(5)

    def disconnect(self):
        self.disconnected = True
        self._ev.set()


def test_manager_start_and_stop():
    client = FakeWatcherClient()
    mgr = DeletedWatcherManager(repository=SimpleNamespace(), build_client=lambda user: client)
    mgr.start_for_user({"bot_user_id": 1})
    # Wait until the watcher is actually blocked in run_until_disconnected.
    assert client.started.wait(2)
    assert mgr.is_running(1)
    mgr.stop_for_user(1)
    for _ in range(100):
        if not mgr.is_running(1) and client.disconnected:
            break
        time.sleep(0.02)
    assert not mgr.is_running(1)
    assert client.disconnected
