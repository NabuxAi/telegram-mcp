import json

from telegram_notebook.vector_search import cosine_similarity, rank_by_cosine


def test_cosine_similarity_basic():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0
    assert round(cosine_similarity([1, 1], [1, 0]), 4) == round(1 / (2 ** 0.5), 4)


def test_cosine_similarity_guards():
    assert cosine_similarity([], [1]) == 0.0
    assert cosine_similarity([1, 2], [1]) == 0.0       # length mismatch
    assert cosine_similarity([0, 0], [1, 1]) == 0.0    # zero vector


def test_rank_by_cosine_orders_and_limits():
    candidates = [
        {"id": "a", "embedding_json": [1.0, 0.0, 0.0]},
        {"id": "b", "embedding_json": [0.0, 1.0, 0.0]},
        {"id": "c", "embedding_json": [0.9, 0.1, 0.0]},
    ]
    ranked = rank_by_cosine([1.0, 0.0, 0.0], candidates, top_k=2)
    assert [c["id"] for c, _ in ranked] == ["a", "c"]  # closest two, b dropped
    assert ranked[0][1] >= ranked[1][1]


def test_rank_by_cosine_parses_json_strings_and_skips_missing():
    candidates = [
        {"id": "a", "embedding_json": json.dumps([1.0, 0.0])},  # JSON string (as stored)
        {"id": "b", "embedding_json": None},                    # no embedding -> skipped
        {"id": "c", "embedding_json": "not json"},              # invalid -> skipped
    ]
    ranked = rank_by_cosine([1.0, 0.0], candidates, top_k=5)
    assert [c["id"] for c, _ in ranked] == ["a"]


def test_rank_by_cosine_empty_inputs():
    assert rank_by_cosine([], [{"embedding_json": [1, 0]}], top_k=3) == []
    assert rank_by_cosine([1, 0], [], top_k=3) == []
    assert rank_by_cosine([1, 0], [{"embedding_json": [1, 0]}], top_k=0) == []
