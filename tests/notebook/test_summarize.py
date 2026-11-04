import asyncio
from types import SimpleNamespace

from telegram_notebook.db import Repository
from telegram_notebook.embeddings import EmbeddingService
from telegram_notebook.pipeline import IngestionPipeline
from telegram_notebook.search import SearchService


def _repo(tmp_path) -> Repository:
    repo = Repository(tmp_path / "store.db")
    repo.init()
    return repo


def _pipeline(repo: Repository) -> IngestionPipeline:
    return IngestionPipeline(
        settings=SimpleNamespace(chunk_size=900, chunk_overlap=120),
        repository=repo,
        transcription=None,
        embeddings=EmbeddingService(provider="openai", api_key=None, model="x"),
    )


def test_build_summary_prompt_includes_sources_and_scope():
    items = [
        {"text": "first note about villas", "channel_title": "Real Estate Channel"},
        {"text": "second note", "channel_url": "https://t.me/x"},
    ]
    prompt = SearchService._build_summary_prompt("Real Estate", items)
    assert "Real Estate" in prompt
    assert "Real Estate Channel" in prompt
    assert "first note about villas" in prompt
    assert "https://t.me/x" in prompt


def test_build_summary_prompt_truncates_long_text():
    items = [{"text": "x" * 5000, "channel_title": "C"}]
    prompt = SearchService._build_summary_prompt("scope", items, per_item_chars=100)
    assert "x" * 100 in prompt
    assert "x" * 101 not in prompt


def test_summarize_returns_message_when_empty():
    svc = SearchService(repository=None, embeddings=EmbeddingService(provider="openai", api_key=None, model="x"))
    assert svc.summarize(scope_label="all", items=[]) == "موردی برای خلاصه‌سازی پیدا نشد."


def test_summary_items_scoping(tmp_path):
    repo = _repo(tmp_path)
    repo.add_rule(owner_id=1, keyword="villa", tag="Real Estate")
    pipe = _pipeline(repo)

    # Two items in the forwarded inbox; one matches the rule -> tagged.
    asyncio.run(pipe.ingest_forwarded_message(owner_id=1, source_label="src", text="a villa by the sea", forward_key=1))
    asyncio.run(pipe.ingest_forwarded_message(owner_id=1, source_label="src", text="an AI agents note", forward_key=2))

    # All items for the owner.
    everything = repo.summary_items(owner_id=1)
    assert len(everything) == 2
    assert {i["text"] for i in everything} == {"a villa by the sea", "an AI agents note"}

    # Scoped by tag.
    real_estate = repo.summary_items(owner_id=1, tag="Real Estate")
    assert len(real_estate) == 1 and real_estate[0]["text"] == "a villa by the sea"

    # Scoped by source (the synthetic forwarded inbox channel).
    from telegram_notebook.pipeline import FORWARDED_INBOX_URL
    by_source = repo.summary_items(owner_id=1, channel_url=FORWARDED_INBOX_URL)
    assert len(by_source) == 2

    # Another user sees nothing.
    assert repo.summary_items(owner_id=2) == []
