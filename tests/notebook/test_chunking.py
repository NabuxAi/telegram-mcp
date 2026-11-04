import pytest

from telegram_notebook.chunking import split_text


def test_empty_returns_no_chunks():
    assert split_text("", 100, 10) == []
    assert split_text("    ", 100, 10) == []


def test_whitespace_is_normalized_into_single_chunk():
    chunks = split_text("  hello   world  ", 100, 10)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].index == 0
    assert chunks[0].start_char == 0


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text("abc", 4, 4)


def test_sliding_window_with_overlap():
    chunks = split_text("abcdefghij", chunk_size=4, overlap=1)
    texts = [c.text for c in chunks]
    # step = size - overlap = 3; windows start at 0, 3, 6 and the last reaches the end
    assert texts == ["abcd", "defg", "ghij"]
    assert chunks[1].text[0] == chunks[0].text[-1]  # one-char overlap
    assert [c.index for c in chunks] == [0, 1, 2]
