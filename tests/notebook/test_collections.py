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


# --- repository ---

def test_collection_crud_and_tags(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    repo.create_collection(owner_id=1, name="Work")
    assert repo.add_collection_tag(owner_id=1, name="Work", tag="AI") is True
    assert repo.add_collection_tag(owner_id=1, name="Work", tag="Finance") is True
    assert repo.add_collection_tag(owner_id=1, name="Missing", tag="X") is False

    cols = repo.list_collections(owner_id=1)
    assert cols == [{"id": cols[0]["id"], "name": "Work", "tags": ["AI", "Finance"]}]
    assert repo.collection_tags(owner_id=1, name="Work") == ["AI", "Finance"]
    assert repo.collection_tags(owner_id=1, name="Nope") is None

    assert repo.remove_collection(owner_id=1, name="Work") is True
    assert repo.list_collections(owner_id=1) == []
    assert repo.remove_collection(owner_id=1, name="Work") is False


def test_collection_scoped_per_user(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    repo.create_collection(owner_id=1, name="Mine")
    repo.create_collection(owner_id=2, name="Mine")
    repo.add_collection_tag(owner_id=1, name="Mine", tag="A")
    assert repo.collection_tags(owner_id=1, name="Mine") == ["A"]
    assert repo.collection_tags(owner_id=2, name="Mine") == []  # separate collection


def test_items_for_tags_union_distinct(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    _item(repo, 1, "ai and finance", ["AI", "Finance"])  # matches via either tag, once
    _item(repo, 1, "only ai", ["AI"])
    _item(repo, 1, "unrelated", ["Other"])
    _item(repo, 2, "other user ai", ["AI"])

    texts = sorted(r["text"] for r in repo.items_for_tags(owner_id=1, tags=["AI", "Finance"]))
    assert texts == ["ai and finance", "only ai"]  # distinct, no "unrelated"/other user
    assert repo.items_for_tags(owner_id=1, tags=[]) == []


# --- handler ---

def _probe(repo):
    p = _Probe()
    api = FakeApi()
    p.services = SimpleNamespace(repository=repo, api=api)
    return p, api


def test_handle_collection_flow(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    _item(repo, 1, "claude is an ai tool", ["AI"])
    p, api = _probe(repo)

    p._handle_collection(10, 1, "new Research")
    assert "created" in api.messages[-1][1]

    p._handle_collection(10, 1, "add Research AI")
    assert "Added" in api.messages[-1][1]

    p._handle_collection(10, 1, "list")
    assert "Research" in api.messages[-1][1] and "AI" in api.messages[-1][1]

    p._handle_collection(10, 1, "show Research")
    msg = api.messages[-1][1]
    assert "Research" in msg and "claude is an ai tool" in msg

    p._handle_collection(10, 1, "remove Research")
    assert "removed" in api.messages[-1][1]


def test_handle_collection_errors(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    p, api = _probe(repo)

    p._handle_collection(10, 1, "add Ghost SomeTag")
    assert "not found" in api.messages[-1][1]
    p._handle_collection(10, 1, "show Ghost")
    assert "not found" in api.messages[-1][1]
    p._handle_collection(10, 1, "list")
    assert "No collections yet" in api.messages[-1][1]
    p._handle_collection(10, 1, "new")
    assert "Usage" in api.messages[-1][1]


def test_extract_collection():
    assert NotebookBot._extract_collection("--collection Work") == ("Work", "")
    assert NotebookBot._extract_collection("foo --collection Work bar") == ("Work", "foo bar")
    assert NotebookBot._extract_collection("no flag here") == (None, "no flag here")
    assert NotebookBot._extract_collection("--collection") == (None, "")  # missing name


def test_summarize_by_collection(tmp_path, monkeypatch):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    _item(repo, 1, "alpha note", ["AI"])
    _item(repo, 1, "beta note", ["Finance"])
    _item(repo, 1, "gamma note", ["Other"])
    repo.create_collection(owner_id=1, name="Work")
    repo.add_collection_tag(owner_id=1, name="Work", tag="AI")
    repo.add_collection_tag(owner_id=1, name="Work", tag="Finance")

    p, api = _probe(repo)
    p.services.settings = SimpleNamespace(gemini_api_key="key", vertex_project_id=None, vertex_region="us-central1")

    captured = {}

    class FakeSearch:
        def summarize(self, *, scope_label, items, **kwargs):
            captured["scope"] = scope_label
            captured["texts"] = sorted(i["text"] for i in items)
            return "the summary"

    monkeypatch.setattr(p, "_search_service_for_user", lambda user: FakeSearch())
    monkeypatch.setattr(p, "_vertex_search_config", lambda user: None)

    p._handle_summarize(10, 1, "--collection Work")
    assert captured["texts"] == ["alpha note", "beta note"]  # union of the collection's tags
    assert "Work" in captured["scope"]
    assert "the summary" in api.messages[-1][1]

    p._handle_summarize(10, 1, "--collection Ghost")
    assert "not found" in api.messages[-1][1]


def test_export_by_collection(tmp_path):
    from pathlib import Path

    repo = Repository(tmp_path / "store.db")
    repo.init()
    _item(repo, 1, "alpha note", ["AI"])
    _item(repo, 1, "skip me", ["Other"])
    repo.create_collection(owner_id=1, name="Work")
    repo.add_collection_tag(owner_id=1, name="Work", tag="AI")

    sent = []

    class DocApi:
        def send_message(self, chat_id=None, text=None, **kwargs):
            sent.append(("msg", text))

        def send_document(self, *, chat_id, document_path, caption=None):
            sent.append(("doc", Path(document_path).read_text(encoding="utf-8"), caption))

    p = _Probe()
    p.services = SimpleNamespace(repository=repo, api=DocApi())
    p._handle_export(10, 1, "--collection Work")
    doc = next(s for s in sent if s[0] == "doc")
    assert "alpha note" in doc[1] and "skip me" not in doc[1]
    assert "Work" in doc[2]
