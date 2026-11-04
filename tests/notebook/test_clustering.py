import json

from telegram_notebook.clustering import (
    build_label_prompt,
    build_topics,
    cluster_embeddings,
    label_cluster,
    parse_topic_label,
    top_terms,
)
from telegram_notebook.db import Repository


def test_build_label_prompt_includes_and_truncates_snippets():
    prompt = build_label_prompt(["x" * 500, "second snippet", ""], max_items=6, max_chars=100)
    assert "x" * 100 in prompt and "x" * 101 not in prompt
    assert "second snippet" in prompt
    assert prompt.rstrip().endswith("Label:")


def test_parse_topic_label_cleans_reply():
    assert parse_topic_label('"Real Estate"') == "Real Estate"
    assert parse_topic_label("\n\n  Video AI Tools \n more") == "Video AI Tools"
    assert parse_topic_label("") == ""
    assert parse_topic_label(None) == ""


def test_label_cluster_uses_injected_generate():
    calls = []

    def fake_generate(prompt: str) -> str:
        calls.append(prompt)
        return "AI Video Tools"

    assert label_cluster(["a snippet about ai video"], generate=fake_generate) == "AI Video Tools"
    assert len(calls) == 1


def test_build_topics_namer_labels_and_falls_back():
    items = [
        {"embedding": [1.0, 0.0], "text": "villa marina villa"},
        {"embedding": [0.97, 0.03], "text": "villa pool"},
    ]
    # namer provides the label
    topics = build_topics(items, threshold=0.8, namer=lambda texts: "Luxury Real Estate")
    assert topics[0]["label"] == "Luxury Real Estate"

    # namer raising -> fall back to keyword-frequency label
    def boom(texts):
        raise RuntimeError("llm down")

    topics = build_topics(items, threshold=0.8, namer=boom)
    assert "villa" in topics[0]["label"]

    # namer returning empty -> fall back too
    topics = build_topics(items, threshold=0.8, namer=lambda texts: "  ")
    assert "villa" in topics[0]["label"]


def test_top_terms_skips_stopwords_and_short_tokens():
    texts = ["The villa and the pool", "villa villa marina", "AI agents and tools"]
    terms = top_terms(texts, k=3)
    assert "villa" in terms  # most frequent significant term
    assert "the" not in terms and "and" not in terms


def test_cluster_embeddings_separates_distinct_groups():
    # Two clearly separated directions in 3D space.
    items = [
        {"embedding": [1.0, 0.0, 0.0], "text": "a"},
        {"embedding": [0.95, 0.05, 0.0], "text": "b"},
        {"embedding": [0.0, 1.0, 0.0], "text": "c"},
        {"embedding": [0.02, 0.98, 0.0], "text": "d"},
    ]
    clusters = cluster_embeddings(items, threshold=0.8)
    assert len(clusters) == 2
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [2, 2]


def test_cluster_respects_max_clusters():
    items = [{"embedding": [float(i == j) for j in range(5)], "text": "x"} for i in range(5)]
    clusters = cluster_embeddings(items, threshold=0.99, max_clusters=2)
    assert len(clusters) == 2
    assert sum(len(c) for c in clusters) == 5  # every item placed


def test_cluster_skips_items_without_embedding():
    items = [{"embedding": None, "text": "x"}, {"embedding": [1.0, 0.0], "text": "y"}]
    clusters = cluster_embeddings(items, threshold=0.5)
    assert len(clusters) == 1 and clusters[0] == [1]


def test_build_topics_labels_and_sorts_by_size():
    items = [
        {"embedding": [1.0, 0.0], "text": "villa marina villa", "channel_title": "RE"},
        {"embedding": [0.97, 0.03], "text": "villa pool", "channel_title": "RE"},
        {"embedding": [0.0, 1.0], "text": "ai agents", "channel_url": "https://t.me/ai"},
    ]
    topics = build_topics(items, threshold=0.8)
    assert len(topics) == 2
    assert topics[0]["size"] == 2  # largest first
    assert "villa" in topics[0]["label"]
    assert topics[0]["sources"] == ["RE"]
    assert topics[0]["sample"]


def test_chunks_with_embeddings_decodes_and_scopes(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    cid = repo.upsert_channel(owner_id=1, telegram_id=1, channel_url="https://t.me/c", title="C", username=None)
    mid = repo.create_or_get_message(channel_id=cid, telegram_message_id=1, message_date=None, message_url="u", caption=None)
    media_id = repo.create_or_get_media(message_id=mid, file_name="t", file_path="", mime_type="text/plain", media_kind="text", duration_seconds=None, file_size_bytes=1)
    repo.replace_chunks(media_item_id=media_id, chunks=[
        {"chunk_index": 0, "text": "embedded chunk", "embedding": [0.1, 0.2, 0.3], "start_char": 0, "end_char": 14},
        {"chunk_index": 1, "text": "no embedding here", "embedding": None, "start_char": 0, "end_char": 17},
    ])

    items = repo.chunks_with_embeddings(owner_id=1)
    assert len(items) == 1  # only the chunk that has an embedding
    assert items[0]["embedding"] == [0.1, 0.2, 0.3]
    assert items[0]["text"] == "embedded chunk"

    # owner isolation
    assert repo.chunks_with_embeddings(owner_id=2) == []
    # stored as JSON bytes in the BLOB
    assert json.loads(_raw_embedding(repo, media_id)) == [0.1, 0.2, 0.3]


def _raw_embedding(repo: Repository, media_id: int):
    import sqlite3
    with sqlite3.connect(repo.path) as conn:
        row = conn.execute("SELECT embedding FROM chunks WHERE media_item_id = ? AND embedding IS NOT NULL", (media_id,)).fetchone()
    return row[0].decode("utf-8")


def test_mcp_list_topics_tool(tmp_path):
    from types import SimpleNamespace

    from telegram_notebook.embeddings import EmbeddingService
    from telegram_notebook.mcp_server import McpServer
    from telegram_notebook.search import SearchService

    repo = Repository(tmp_path / "store.db")
    repo.init()
    server = McpServer(
        repository=repo,
        search_service=SearchService(repo, EmbeddingService(provider="openai", api_key=None, model="x")),
        settings=SimpleNamespace(gemini_api_key=None, vertex_project_id=None, vertex_region="us-central1"),
        owner_id=1,
    )

    def call():
        return server.handle_request({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "list_topics", "arguments": {}},
        })["result"]["content"][0]["text"]

    # advertised in tools/list
    listed = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert "list_topics" in {t["name"] for t in listed["result"]["tools"]}

    # empty archive -> friendly message
    assert "No embedded content" in call()

    # seed an embedded chunk -> a topic is produced
    cid = repo.upsert_channel(owner_id=1, telegram_id=1, channel_url="https://t.me/c", title="C", username=None)
    mid = repo.create_or_get_message(channel_id=cid, telegram_message_id=1, message_date=None, message_url="u", caption=None)
    media_id = repo.create_or_get_media(message_id=mid, file_name="t", file_path="", mime_type="text/plain", media_kind="text", duration_seconds=None, file_size_bytes=1)
    repo.replace_chunks(media_item_id=media_id, chunks=[
        {"chunk_index": 0, "text": "villa marina pool", "embedding": [1.0, 0.0], "start_char": 0, "end_char": 5},
    ])
    assert "villa" in call()
