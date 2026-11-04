from telegram_notebook.media import detect_media_kind


def test_detect_by_extension():
    assert detect_media_kind("clip.mp4", None) == "video"
    assert detect_media_kind("song.mp3", None) == "audio"


def test_detect_by_mime():
    assert detect_media_kind(None, "video/mp4") == "video"
    assert detect_media_kind(None, "audio/mpeg") == "audio"


def test_non_media_returns_none():
    assert detect_media_kind("notes.txt", None) is None
    assert detect_media_kind("doc.pdf", "application/pdf") is None
