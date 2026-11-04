from telegram_notebook.media import route_media


def test_route_media_audio_video():
    assert route_media("audio", None) == "transcribe"
    assert route_media("video", "video/mp4") == "transcribe"


def test_route_media_image():
    assert route_media("image", None) == "extract"
    assert route_media("image", "image/png") == "extract"


def test_route_media_documents_by_mime_and_extension():
    assert route_media("document", "application/pdf") == "extract"
    assert route_media("document", "image/png") == "extract"
    assert route_media("document", "audio/ogg") == "transcribe"
    assert route_media("document", "video/mp4") == "transcribe"
    # office formats are detected by extension even with a generic/empty mime
    assert route_media("document", "", "report.docx") == "office"
    assert route_media("document", "application/octet-stream", "budget.xlsx") == "office"


def test_route_media_unsupported_and_text():
    assert route_media("document", "application/zip") is None
    assert route_media("text", None) is None
    assert route_media(None, None) is None
