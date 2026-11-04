"""Tokenized keyword search: multi-word / question queries must still match."""

from __future__ import annotations

from telegram_notebook.db import Repository
from telegram_notebook.keyword_search import query_terms

# --- query_terms (pure) ---

def test_query_terms_splits_and_strips_punctuation():
    assert query_terms("فرشاد کیه؟") == ["فرشاد", "کیه"]
    assert query_terms("تیم خوب میخریدن؟") == ["تیم", "خوب", "میخریدن"]
    assert query_terms("who is Farshad?") == ["who", "is", "farshad"]


def test_query_terms_drops_short_and_dedupes():
    assert query_terms("a I am am") == ["am"]          # 1-char dropped, dup removed
    assert query_terms("?! . ،") == []                 # punctuation only -> no terms


def test_query_terms_caps_count():
    many = " ".join(f"term{i}" for i in range(50))
    assert len(query_terms(many, max_terms=5)) == 5


# --- keyword_candidates integration ---

def _seed(repo, *, owner_id, message_url, text):
    cid = repo.upsert_channel(owner_id=owner_id, telegram_id=abs(hash(message_url)) % 9999,
                              channel_url="https://t.me/c", title="C", username=None)
    mid = repo.create_or_get_message(channel_id=cid, telegram_message_id=abs(hash(message_url)) % 9999,
                                     message_date="2026-06-01T00:00:00", message_url=message_url, caption=None)
    media_id = repo.create_or_get_media(message_id=mid, file_name="f", file_path="", mime_type="text/plain",
                                        media_kind="text", duration_seconds=None, file_size_bytes=len(text))
    repo.replace_chunks(media_item_id=media_id, chunks=[
        {"chunk_index": 0, "text": text, "embedding": None, "start_char": 0, "end_char": len(text)},
    ])


def test_question_query_matches_individual_words(tmp_path):
    """Reproduces the reported bug: '/search فرشاد کیه؟' returned nothing."""
    repo = Repository(tmp_path / "store.db")
    repo.init()
    _seed(repo, owner_id=1, message_url="u1", text="از طرف فرشاد پیام میدم بهت")
    _seed(repo, owner_id=1, message_url="u2", text="سلام و درود خوبی عزیزم")

    # whole-phrase substring never appears, but the term "فرشاد" does
    hits = repo.keyword_candidates(owner_id=1, query="فرشاد کیه؟", top_k=5, channel_url=None)
    assert [h["message_url"] for h in hits] == ["u1"]


def test_ranks_by_number_of_matching_terms(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    _seed(repo, owner_id=1, message_url="both", text="تیم خوب میخریدن کوفتی رو")
    _seed(repo, owner_id=1, message_url="one", text="هوا خیلی خوب است")

    hits = repo.keyword_candidates(owner_id=1, query="تیم خوب میخریدن؟", top_k=5, channel_url=None)
    # the chunk hitting more query terms ranks first
    assert hits[0]["message_url"] == "both"
    assert {h["message_url"] for h in hits} == {"both", "one"}


def test_single_word_query_unchanged(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    _seed(repo, owner_id=1, message_url="u1", text="a note about townhouses near the marina")
    assert len(repo.keyword_candidates(owner_id=1, query="townhouses", top_k=5, channel_url=None)) == 1
    assert repo.keyword_candidates(owner_id=1, query="nonexistent", top_k=5, channel_url=None) == []
