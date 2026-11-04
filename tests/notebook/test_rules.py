import asyncio
from types import SimpleNamespace

from telegram_notebook.bot import NotebookBot
from telegram_notebook.db import Repository
from telegram_notebook.embeddings import EmbeddingService
from telegram_notebook.pipeline import IngestionPipeline
from telegram_notebook.rules import match_tags

# --- pure matcher ---

def test_match_tags_is_case_insensitive_substring():
    rules = [
        {"keyword": "Claude", "tag": "AI Tools"},
        {"keyword": "Al Mouj", "tag": "Real Estate"},
    ]
    assert match_tags("I love claude and codex", rules) == {"AI Tools"}
    assert match_tags("a villa in AL MOUJ marina", rules) == {"Real Estate"}
    assert match_tags("claude tours al mouj", rules) == {"AI Tools", "Real Estate"}
    assert match_tags("nothing relevant", rules) == set()
    assert match_tags("", rules) == set()


def test_parse_rule_add():
    assert NotebookBot._parse_rule_add("Claude -> AI Tools") == ("Claude", "AI Tools")
    assert NotebookBot._parse_rule_add("  golden visa  ->  Oman Visa ") == ("golden visa", "Oman Visa")
    assert NotebookBot._parse_rule_add("missing arrow") is None
    assert NotebookBot._parse_rule_add("-> only tag") is None
    assert NotebookBot._parse_rule_add("only keyword ->") is None


# --- repository storage ---

def _repo(tmp_path) -> Repository:
    repo = Repository(tmp_path / "store.db")
    repo.init()
    return repo


def test_rule_crud_and_uniqueness(tmp_path):
    repo = _repo(tmp_path)
    rid = repo.add_rule(owner_id=1, keyword="claude", tag="AI Tools")
    assert rid > 0
    # duplicate (same owner/keyword/tag) is ignored, returns same id
    assert repo.add_rule(owner_id=1, keyword="claude", tag="AI Tools") == rid
    assert len(repo.list_rules(owner_id=1)) == 1
    # rules are per-owner
    assert repo.list_rules(owner_id=2) == []
    assert repo.remove_rule(owner_id=1, rule_id=rid) is True
    assert repo.remove_rule(owner_id=1, rule_id=rid) is False
    assert repo.list_rules(owner_id=1) == []


def test_tag_storage_and_listing(tmp_path):
    repo = _repo(tmp_path)
    repo.tag_media(owner_id=1, media_item_id=10, tags={"AI Tools", "News"})
    repo.tag_media(owner_id=1, media_item_id=11, tags=["AI Tools"])
    repo.tag_media(owner_id=1, media_item_id=10, tags={"AI Tools"})  # idempotent

    tags = {t["tag"]: t["count"] for t in repo.list_tags(owner_id=1)}
    assert tags == {"AI Tools": 2, "News": 1}
    assert repo.media_ids_for_tag(owner_id=1, tag="AI Tools") == {10, 11}
    assert repo.media_ids_for_tag(owner_id=2, tag="AI Tools") == set()

    repo.clear_tags(owner_id=1)
    assert repo.list_tags(owner_id=1) == []


# --- end-to-end auto-tagging on ingest + tag-filtered keyword search ---

def _pipeline(repo: Repository) -> IngestionPipeline:
    return IngestionPipeline(
        settings=SimpleNamespace(chunk_size=900, chunk_overlap=120),
        repository=repo,
        transcription=None,
        embeddings=EmbeddingService(provider="openai", api_key=None, model="x"),
    )


def test_forwarded_ingest_auto_tags_and_search_filters_by_tag(tmp_path):
    repo = _repo(tmp_path)
    repo.add_rule(owner_id=1, keyword="townhouse", tag="Real Estate")
    pipe = _pipeline(repo)

    asyncio.run(pipe.ingest_forwarded_message(
        owner_id=1, source_label="src", text="a lovely townhouse near the marina", forward_key=1,
    ))
    asyncio.run(pipe.ingest_forwarded_message(
        owner_id=1, source_label="src", text="a note about AI agents", forward_key=2,
    ))

    # The matching item is tagged; the other is not.
    assert {t["tag"]: t["count"] for t in repo.list_tags(owner_id=1)} == {"Real Estate": 1}

    # Tag-filtered keyword search only returns the tagged item.
    tagged = repo.keyword_candidates(owner_id=1, query="a", top_k=10, channel_url=None, tag="Real Estate")
    assert len(tagged) == 1 and "townhouse" in tagged[0]["chunk_text"]

    # Without the filter, both items are searchable.
    assert len(repo.keyword_candidates(owner_id=1, query="a", top_k=10, channel_url=None)) == 2


def test_retag_via_media_texts(tmp_path):
    """A rule added after ingestion can backfill tags from stored transcripts."""
    repo = _repo(tmp_path)
    pipe = _pipeline(repo)
    asyncio.run(pipe.ingest_forwarded_message(
        owner_id=1, source_label="src", text="claude is a great assistant", forward_key=1,
    ))
    # No rule at ingest time => no tags.
    assert repo.list_tags(owner_id=1) == []

    repo.add_rule(owner_id=1, keyword="claude", tag="AI Tools")
    rules = repo.list_rules(owner_id=1)
    for item in repo.media_texts(owner_id=1):
        repo.tag_media(owner_id=1, media_item_id=item["media_item_id"], tags=match_tags(item["text"], rules))

    assert {t["tag"]: t["count"] for t in repo.list_tags(owner_id=1)} == {"AI Tools": 1}
