"""SearchService local semantic search over stored chunk embeddings (no network)."""

from __future__ import annotations

from telegram_notebook.db import Repository
from telegram_notebook.search import SearchService


class FakeEmbeddings:
    """Duck-typed EmbeddingService: enabled, with a fixed text->vector mapping."""

    api_key = None

    def __init__(self, mapping, *, enabled=True):
        self.enabled = enabled
        self.mapping = mapping

    def embed(self, text, **kwargs):
        return self.mapping.get(text)


def _seed_chunk(repo, *, owner_id, channel_url, title, message_url, text, embedding, tags=None):
    cid = repo.upsert_channel(
        owner_id=owner_id, telegram_id=abs(hash(channel_url)) % 9999,
        channel_url=channel_url, title=title, username=None,
    )
    mid = repo.create_or_get_message(
        channel_id=cid, telegram_message_id=abs(hash(message_url)) % 9999,
        message_date="2026-06-01T00:00:00", message_url=message_url, caption=None,
    )
    media_id = repo.create_or_get_media(
        message_id=mid, file_name="f", file_path="", mime_type="text/plain",
        media_kind="text", duration_seconds=None, file_size_bytes=len(text),
    )
    repo.replace_chunks(media_item_id=media_id, chunks=[
        {"chunk_index": 0, "text": text, "embedding": embedding, "start_char": 0, "end_char": len(text)},
    ])
    if tags:
        repo.tag_media(owner_id=owner_id, media_item_id=media_id, tags=set(tags))
    return media_id


def _repo_with_two_docs(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    _seed_chunk(repo, owner_id=1, channel_url="https://t.me/a", title="A", message_url="u_pets",
                text="cats dogs and pets", embedding=[1.0, 0.0, 0.0], tags=["pets"])
    _seed_chunk(repo, owner_id=1, channel_url="https://t.me/a", title="A", message_url="u_money",
                text="stocks finance and money", embedding=[0.0, 1.0, 0.0], tags=["finance"])
    return repo


def test_local_vector_search_ranks_semantically(tmp_path):
    repo = _repo_with_two_docs(tmp_path)
    svc = SearchService(repo, FakeEmbeddings({"pets": [1.0, 0.0, 0.0]}))
    results = svc.search(owner_id=1, query="pets", top_k=2)
    assert [r.message_url for r in results] == ["u_pets", "u_money"]  # semantic order
    assert results[0].score >= results[1].score
    assert results[0].chunk_text == "cats dogs and pets"


def test_local_vector_search_respects_top_k(tmp_path):
    repo = _repo_with_two_docs(tmp_path)
    svc = SearchService(repo, FakeEmbeddings({"pets": [1.0, 0.0, 0.0]}))
    assert len(svc.search(owner_id=1, query="pets", top_k=1)) == 1


def test_local_vector_search_channel_filter(tmp_path):
    repo = _repo_with_two_docs(tmp_path)
    svc = SearchService(repo, FakeEmbeddings({"pets": [1.0, 0.0, 0.0]}))
    # a channel with no embedded chunks yields nothing (and keyword finds nothing either)
    assert svc.search(owner_id=1, query="pets", channel_url="https://t.me/other", top_k=5) == []
    hits = svc.search(owner_id=1, query="pets", channel_url="https://t.me/a", top_k=5)
    assert {r.message_url for r in hits} == {"u_pets", "u_money"}


def test_local_vector_search_tag_filter(tmp_path):
    repo = _repo_with_two_docs(tmp_path)
    svc = SearchService(repo, FakeEmbeddings({"anything": [1.0, 0.0, 0.0]}))
    # the tag restricts the candidate pool to the finance doc, regardless of vector
    results = svc.search(owner_id=1, query="anything", tag="finance", top_k=5)
    assert [r.message_url for r in results] == ["u_money"]


def test_falls_back_to_keyword_when_no_query_vector(tmp_path):
    repo = _repo_with_two_docs(tmp_path)
    # "dogs" isn't in the mapping -> embed returns None -> lexical fallback over text
    svc = SearchService(repo, FakeEmbeddings({"pets": [1.0, 0.0, 0.0]}))
    results = svc.search(owner_id=1, query="dogs", top_k=5)
    assert [r.message_url for r in results] == ["u_pets"]


def test_disabled_embeddings_uses_keyword(tmp_path):
    repo = _repo_with_two_docs(tmp_path)
    svc = SearchService(repo, FakeEmbeddings({}, enabled=False))
    results = svc.search(owner_id=1, query="finance", top_k=5)
    assert [r.message_url for r in results] == ["u_money"]


def test_isolated_per_owner(tmp_path):
    repo = _repo_with_two_docs(tmp_path)
    svc = SearchService(repo, FakeEmbeddings({"pets": [1.0, 0.0, 0.0]}))
    assert svc.search(owner_id=2, query="pets", top_k=5) == []
