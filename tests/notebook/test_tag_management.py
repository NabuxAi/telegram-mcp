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


def _item(repo, owner_id, text, tags):
    cid = repo.upsert_channel(owner_id=owner_id, telegram_id=1, channel_url="https://t.me/c", title="C", username=None)
    mid = repo.create_or_get_message(channel_id=cid, telegram_message_id=hash(text) % 9999, message_date="2026-06-01T00:00:00", message_url="u", caption=None)
    media_id = repo.create_or_get_media(message_id=mid, file_name="t", file_path="", mime_type="text/plain", media_kind="text", duration_seconds=None, file_size_bytes=1)
    repo.mark_media_transcribed(media_item_id=media_id, transcript_text=text)
    repo.tag_media(owner_id=owner_id, media_item_id=media_id, tags=set(tags))
    return media_id


def _tagcounts(repo, owner_id):
    return {t["tag"]: t["count"] for t in repo.list_tags(owner_id=owner_id)}


def test_rename_tag(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    _item(repo, 1, "a", ["Old"])
    _item(repo, 1, "b", ["Old"])
    moved = repo.rename_tag(owner_id=1, old_tag="Old", new_tag="New")
    assert moved == 2
    assert _tagcounts(repo, 1) == {"New": 2}


def test_rename_tag_merges_into_existing(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    m1 = _item(repo, 1, "a", ["A", "B"])  # has both A and B
    _item(repo, 1, "b", ["A"])            # only A
    # rename A -> B; item m1 already has B (PK collision handled), item2 becomes B
    moved = repo.rename_tag(owner_id=1, old_tag="A", new_tag="B")
    assert moved == 2
    counts = _tagcounts(repo, 1)
    assert counts == {"B": 2}
    # both items end up under B (set dedupes the collision cleanly)
    ids = repo.media_ids_for_tag(owner_id=1, tag="B")
    assert m1 in ids and len(ids) == 2


def test_delete_tag(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    _item(repo, 1, "a", ["Keep", "Drop"])
    _item(repo, 1, "b", ["Drop"])
    removed = repo.delete_tag(owner_id=1, tag="Drop")
    assert removed == 2
    assert _tagcounts(repo, 1) == {"Keep": 1}


def test_tag_isolation_between_users(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    _item(repo, 1, "a", ["Shared"])
    _item(repo, 2, "b", ["Shared"])
    repo.rename_tag(owner_id=1, old_tag="Shared", new_tag="Renamed")
    assert _tagcounts(repo, 1) == {"Renamed": 1}
    assert _tagcounts(repo, 2) == {"Shared": 1}  # other user untouched


def _probe(repo):
    p = _Probe()
    api = FakeApi()
    p.services = SimpleNamespace(repository=repo, api=api)
    return p, api


def test_handle_tag_rename_and_delete(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    _item(repo, 1, "a", ["Old"])
    p, api = _probe(repo)

    p._handle_tag(10, 1, "rename Old -> Fresh")
    assert "Renamed" in api.messages[-1][1]
    assert _tagcounts(repo, 1) == {"Fresh": 1}

    p._handle_tag(10, 1, "delete Fresh")
    assert "Deleted" in api.messages[-1][1]
    assert _tagcounts(repo, 1) == {}


def test_handle_tag_usage_and_missing(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    p, api = _probe(repo)
    p._handle_tag(10, 1, "")
    assert "Usage" in api.messages[-1][1]
    p._handle_tag(10, 1, "rename OnlyOld")
    assert "Usage" in api.messages[-1][1]
    p._handle_tag(10, 1, "delete Nope")
    assert "No items" in api.messages[-1][1]
