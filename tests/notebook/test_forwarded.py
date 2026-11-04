import asyncio
from types import SimpleNamespace

from telegram_notebook.bot import NotebookBot
from telegram_notebook.db import Repository
from telegram_notebook.embeddings import EmbeddingService
from telegram_notebook.pipeline import FORWARDED_INBOX_URL, IngestionPipeline

# --- Forward-detection / attribution helpers (pure functions) ---

def test_is_forwarded():
    assert NotebookBot._is_forwarded({"forward_origin": {"type": "user"}}) is True
    assert NotebookBot._is_forwarded({"forward_sender_name": "Anon"}) is True
    assert NotebookBot._is_forwarded({"forward_from": {"id": 1}}) is True
    assert NotebookBot._is_forwarded({"text": "hello"}) is False


def test_source_label_variants():
    assert NotebookBot._forward_source_label(
        {"forward_origin": {"type": "channel", "chat": {"title": "AI News"}}}
    ) == "AI News"
    assert NotebookBot._forward_source_label(
        {"forward_origin": {"type": "user", "sender_user": {"first_name": "Ada", "last_name": "L"}}}
    ) == "Ada L"
    assert NotebookBot._forward_source_label(
        {"forward_origin": {"type": "hidden_user", "sender_user_name": "Secret"}}
    ) == "Secret"
    # legacy fields
    assert NotebookBot._forward_source_label(
        {"forward_from_chat": {"title": "Old Channel"}}
    ) == "Old Channel"
    assert NotebookBot._forward_source_label({"forward_sender_name": "Anon"}) == "Anon"


def test_forward_message_url():
    assert NotebookBot._forward_message_url(
        {"forward_origin": {"type": "channel", "chat": {"username": "ainews"}, "message_id": 42}}
    ) == "https://t.me/ainews/42"
    # private chat / hidden user has no public link
    assert NotebookBot._forward_message_url(
        {"forward_origin": {"type": "hidden_user", "sender_user_name": "x"}}
    ) is None


def test_forward_media_tag():
    assert NotebookBot._forward_media_tag({"document": {"file_name": "report.pdf"}}) == "[Forwarded document: report.pdf]"
    assert NotebookBot._forward_media_tag({"photo": []}) == "[Forwarded photo]"
    assert NotebookBot._forward_media_tag({"text": "no media"}) is None


# --- Forwarded-inbox ingestion (integration over a real SQLite repo) ---

def _pipeline(repo: Repository) -> IngestionPipeline:
    return IngestionPipeline(
        settings=SimpleNamespace(chunk_size=900, chunk_overlap=120),
        repository=repo,
        transcription=None,
        embeddings=EmbeddingService(provider="openai", api_key=None, model="x"),  # disabled, no network
    )


def test_forwarded_message_is_stored_and_searchable(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    pipe = _pipeline(repo)

    stored = asyncio.run(pipe.ingest_forwarded_message(
        owner_id=1, source_label="AI News", text="a note about forwarded townhouses", forward_key=555,
    ))
    assert stored is True

    # It lands under the per-user Forwarded Inbox channel and is keyword-searchable.
    channels = repo.list_channels(owner_id=1)
    assert any(c["channel_url"] == FORWARDED_INBOX_URL for c in channels)
    rows = repo.keyword_candidates(owner_id=1, query="townhouses", top_k=5, channel_url=None)
    assert len(rows) == 1 and "townhouses" in rows[0]["chunk_text"]

    # Idempotent on forward_key.
    again = asyncio.run(pipe.ingest_forwarded_message(
        owner_id=1, source_label="AI News", text="a note about forwarded townhouses", forward_key=555,
    ))
    assert again is False
    assert len(repo.keyword_candidates(owner_id=1, query="townhouses", top_k=5, channel_url=None)) == 1

    # Another user does not see it.
    assert repo.keyword_candidates(owner_id=2, query="townhouses", top_k=5, channel_url=None) == []


def test_forwarded_inbox_is_per_user(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    pipe = _pipeline(repo)
    asyncio.run(pipe.ingest_forwarded_message(owner_id=1, source_label="src", text="user one secret", forward_key=1))
    asyncio.run(pipe.ingest_forwarded_message(owner_id=2, source_label="src", text="user two secret", forward_key=1))

    r1 = repo.keyword_candidates(owner_id=1, query="secret", top_k=5, channel_url=None)
    r2 = repo.keyword_candidates(owner_id=2, query="secret", top_k=5, channel_url=None)
    assert len(r1) == 1 and r1[0]["chunk_text"] == "user one secret"
    assert len(r2) == 1 and r2[0]["chunk_text"] == "user two secret"
