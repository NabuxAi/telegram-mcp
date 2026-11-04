import zipfile
from pathlib import Path

from telegram_notebook.bot import NotebookBot
from telegram_notebook.bot_api import TelegramBotApi


class _Probe(NotebookBot):
    """NotebookBot with the heavy __init__ skipped — only the pure media methods are exercised."""
    def __init__(self):
        pass


class FakeTranscription:
    enabled = True

    def __init__(self, out="transcript text"):
        self.out = out
        self.calls = []

    def transcribe_media(self, path, work_dir):
        self.calls.append(path)
        return self.out


class FakeExtraction:
    enabled = True

    def __init__(self, out="ocr text"):
        self.out = out
        self.calls = []

    def extract(self, path):
        self.calls.append(path)
        return self.out


# --- file selection ---

def test_forward_file_ref_picks_largest_photo():
    msg = {"photo": [{"file_id": "small"}, {"file_id": "big"}]}
    ref = NotebookBot._forward_file_ref(msg)
    assert ref["kind"] == "photo" and ref["file_id"] == "big"


def test_forward_file_ref_document_and_voice():
    doc = NotebookBot._forward_file_ref({"document": {"file_id": "d1", "file_name": "r.pdf", "mime_type": "application/pdf"}})
    assert doc == {"file_id": "d1", "kind": "document", "file_name": "r.pdf", "mime_type": "application/pdf"}
    voice = NotebookBot._forward_file_ref({"voice": {"file_id": "v1"}})
    assert voice["kind"] == "voice" and voice["file_id"] == "v1"
    assert NotebookBot._forward_file_ref({"text": "no media"}) is None


# --- routing ---

def test_media_route():
    assert NotebookBot._media_route("voice", None) == "transcribe"
    assert NotebookBot._media_route("video", "video/mp4") == "transcribe"
    assert NotebookBot._media_route("photo", "image/jpeg") == "extract"
    assert NotebookBot._media_route("document", "application/pdf") == "extract"
    assert NotebookBot._media_route("document", "image/png") == "extract"
    assert NotebookBot._media_route("document", "audio/ogg") == "transcribe"
    assert NotebookBot._media_route("document", "application/zip") is None


def test_media_route_office_documents():
    docx_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert NotebookBot._media_route("document", docx_mime) == "office"
    # detected by extension even when the mime type is generic/empty
    assert NotebookBot._media_route("document", "", "report.docx") == "office"
    assert NotebookBot._media_route("document", "application/octet-stream", "budget.xlsx") == "office"
    assert NotebookBot._media_route("document", "", "notes.txt") is None


# --- orchestration ---

def _download_to(path: Path):
    def download(file_id, file_name):
        return path
    return download


def test_process_forwarded_media_transcribes_audio(tmp_path):
    probe = _Probe()
    tr = FakeTranscription("hello world")
    f = tmp_path / "voice.ogg"
    f.write_bytes(b"x")
    out = probe._process_forwarded_media(
        {"voice": {"file_id": "v1"}}, transcription=tr, extraction=None, download=_download_to(f)
    )
    assert out == "hello world"
    assert tr.calls == [f]


def test_process_forwarded_media_extracts_image(tmp_path):
    probe = _Probe()
    ex = FakeExtraction("OCR result")
    f = tmp_path / "photo.jpg"
    f.write_bytes(b"x")
    out = probe._process_forwarded_media(
        {"photo": [{"file_id": "p1"}]}, transcription=None, extraction=ex, download=_download_to(f)
    )
    assert out == "OCR result"
    assert ex.calls == [f]


def test_process_forwarded_media_extracts_office_without_service(tmp_path):
    probe = _Probe()
    f = tmp_path / "report.docx"
    w_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    with zipfile.ZipFile(f, "w") as zf:
        zf.writestr(
            "word/document.xml",
            f'<w:document xmlns:w="{w_ns}"><w:body>'
            f"<w:p><w:r><w:t>Quarterly report</w:t></w:r></w:p>"
            f"</w:body></w:document>",
        )
    msg = {"document": {"file_id": "d1", "file_name": "report.docx", "mime_type": ""}}
    # No transcription or extraction service is provided — office parsing is local.
    out = probe._process_forwarded_media(
        msg, transcription=None, extraction=None, download=_download_to(f)
    )
    assert out == "Quarterly report"


def test_process_forwarded_media_skips_when_service_disabled(tmp_path):
    probe = _Probe()
    ex = FakeExtraction()
    ex.enabled = False
    called = []
    out = probe._process_forwarded_media(
        {"photo": [{"file_id": "p1"}]}, transcription=None, extraction=ex,
        download=lambda *a: called.append(a),
    )
    assert out is None
    assert called == []  # never downloads when the service is disabled


def test_process_forwarded_media_no_route_or_failed_download(tmp_path):
    probe = _Probe()
    # unsupported document type -> no route
    assert probe._process_forwarded_media(
        {"document": {"file_id": "d", "mime_type": "application/zip"}},
        transcription=FakeTranscription(), extraction=FakeExtraction(), download=_download_to(tmp_path / "x"),
    ) is None
    # download returns None
    assert probe._process_forwarded_media(
        {"voice": {"file_id": "v"}}, transcription=FakeTranscription(), extraction=None,
        download=lambda *a: None,
    ) is None


def test_process_forwarded_media_swallows_service_error(tmp_path):
    probe = _Probe()

    class Boom:
        enabled = True

        def transcribe_media(self, path, work_dir):
            raise RuntimeError("boom")

    f = tmp_path / "v.ogg"
    f.write_bytes(b"x")
    assert probe._process_forwarded_media(
        {"voice": {"file_id": "v"}}, transcription=Boom(), extraction=None, download=_download_to(f)
    ) is None


# --- bot API file URL ---

def test_bot_api_file_base_url():
    api = TelegramBotApi("123:ABC")
    assert api.file_base_url == "https://api.telegram.org/file/bot123:ABC"
