"""End-to-end test of the full channel-import path with a fake Telethon client.

Exercises pipeline.ingest_channel over text, photo (OCR), DOCX (local extraction),
and audio (transcription) messages — plus auto-forward, resume cursor, and
cancellation — with no network, no real session, and no event loop tricks. The
``client`` is injected, so the download → extract → tag → forward flow runs for real.
"""

from __future__ import annotations

import asyncio
import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

from telegram_notebook.db import Repository
from telegram_notebook.embeddings import EmbeddingService
from telegram_notebook.pipeline import IngestionPipeline

OFFICE_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _docx_bytes(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "word/document.xml",
            f'<w:document xmlns:w="{W_NS}"><w:body>'
            f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>",
        )
    return buf.getvalue()


class _File:
    def __init__(self, name, mime_type, size=16):
        self.name = name
        self.mime_type = mime_type
        self.size = size


class _Msg:
    def __init__(self, *, id, text="", date="2026-06-01T00:00:00+00:00", file=None,
                 content=b"x", download_name=None, photo=False, video=False,
                 audio=False, voice=False):
        self.id = id
        self.message = text
        self.date = SimpleNamespace(isoformat=lambda d=date: d)
        self.file = file
        self.photo = photo or None
        self.video = video or None
        self.audio = audio or None
        self.voice = voice or None
        self._content = content
        self._download_name = download_name or (file.name if file else f"msg{id}.bin")


class _Entity:
    id = 4242
    username = "demo"
    title = "Demo Channel"


class FakeTelethonClient:
    def __init__(self, messages):
        self._messages = messages  # newest-first, like Telethon
        self.downloads = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_entity(self, channel_url):
        return _Entity()

    async def iter_messages(self, entity, limit=None, min_id=0):
        count = 0
        for msg in self._messages:
            if min_id and msg.id <= min_id:
                continue
            if limit is not None and count >= limit:
                break
            count += 1
            yield msg

    async def download_media(self, message, file=None):
        self.downloads += 1
        target = Path(file)
        target.mkdir(parents=True, exist_ok=True)
        path = target / message._download_name
        path.write_bytes(message._content)
        return str(path)


class FakeTranscription:
    enabled = True

    def transcribe_media(self, path, work_dir):
        return f"TRANSCRIPT[{Path(path).name}]"


class FakeExtraction:
    enabled = True

    def extract(self, path):
        return f"OCR[{Path(path).name}]"


def _messages():
    # newest-first (descending id), mirroring Telethon iteration order
    return [
        _Msg(id=5, text="townhouse near Al Mouj for sale"),  # text
        _Msg(id=4, photo=True, download_name="photo4.jpg", content=b"\xff\xd8"),  # image -> OCR
        _Msg(id=3, file=_File("report.docx", OFFICE_DOCX_MIME),
             content=_docx_bytes("quarterly numbers about claude")),  # office
        _Msg(id=2, file=_File("talk.mp3", "audio/mpeg"), content=b"ID3"),  # audio -> transcript
        _Msg(id=1, photo=True, download_name="photo1.jpg", content=b"\xff\xd8"),  # image -> OCR
    ]


def _pipeline(repo, settings, *, auto_forward=None):
    return IngestionPipeline(
        settings=settings,
        repository=repo,
        transcription=FakeTranscription(),
        embeddings=EmbeddingService(provider="openai", api_key=None, model="x"),  # disabled
        extraction=FakeExtraction(),
        auto_forward=auto_forward,
    )


def _settings(tmp_path):
    return SimpleNamespace(chunk_size=900, chunk_overlap=120, media_dir=tmp_path / "media")


def test_full_import_processes_every_media_kind(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    fake = FakeTelethonClient(_messages())
    pipe = _pipeline(repo, _settings(tmp_path))

    stats = asyncio.run(pipe.ingest_channel(
        owner_id=7, channel_url="https://t.me/demo", limit=None, client=fake,
    ))

    assert stats.processed_messages == 5
    assert stats.processed_media == 5  # text + 2 images + docx + audio
    assert fake.downloads == 4  # everything but the text-only message is downloaded

    def hits(query):
        return repo.keyword_candidates(owner_id=7, query=query, top_k=10, channel_url=None)

    assert len(hits("townhouse")) == 1          # plain text message
    assert len(hits("OCR")) == 2                # both photos OCR'd
    assert len(hits("TRANSCRIPT")) == 1         # audio transcribed
    assert len(hits("quarterly")) == 1          # DOCX parsed locally
    # other users see nothing
    assert hits("townhouse") and repo.keyword_candidates(owner_id=8, query="townhouse", top_k=10, channel_url=None) == []


def test_full_import_auto_forwards_tagged_items(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    repo.add_rule(owner_id=7, keyword="claude", tag="AI", kind="keyword", created_at="2026-06-01T00:00:00")

    forwarded: list[tuple] = []

    def auto_forward(label, tags, text, url):
        forwarded.append((label, sorted(tags), text, url))

    pipe = _pipeline(repo, _settings(tmp_path), auto_forward=auto_forward)
    asyncio.run(pipe.ingest_channel(
        owner_id=7, channel_url="https://t.me/demo", limit=None, client=FakeTelethonClient(_messages()),
    ))

    # Only the DOCX item contains "claude", so exactly one tagged item is forwarded.
    assert len(forwarded) == 1
    label, tags, text, url = forwarded[0]
    assert label == "Demo Channel"
    assert tags == ["AI"]
    assert "quarterly" in text
    assert url == "https://t.me/demo/3"


def test_import_respects_resume_cursor_and_cancel(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()

    # resume_from is the highest id already processed -> only newer ids are fetched
    stats = asyncio.run(_pipeline(repo, _settings(tmp_path)).ingest_channel(
        owner_id=7, channel_url="https://t.me/demo", limit=None, resume_from=3,
        client=FakeTelethonClient(_messages()),
    ))
    assert stats.processed_messages == 2  # ids 4 and 5 only

    # cancellation stops the loop after the first item
    repo2 = Repository(tmp_path / "store2.db")
    repo2.init()
    state = {"n": 0}

    def should_cancel():
        state["n"] += 1
        return state["n"] > 1  # process the first item, then stop

    stats2 = asyncio.run(_pipeline(repo2, _settings(tmp_path)).ingest_channel(
        owner_id=7, channel_url="https://t.me/demo", limit=None,
        should_cancel=should_cancel, client=FakeTelethonClient(_messages()),
    ))
    assert stats2.processed_media == 1


def test_import_without_extractors_indexes_caption_only(tmp_path):
    """No transcription/extraction services: media files are skipped, captions kept."""
    repo = Repository(tmp_path / "store.db")
    repo.init()
    pipe = IngestionPipeline(
        settings=_settings(tmp_path),
        repository=repo,
        transcription=None,
        embeddings=EmbeddingService(provider="openai", api_key=None, model="x"),
        extraction=None,
    )
    msgs = [
        _Msg(id=2, file=_File("clip.mp3", "audio/mpeg"), text="captioned audio", content=b"ID3"),
        _Msg(id=1, photo=True, download_name="p.jpg", content=b"\xff\xd8"),  # no caption, no key
    ]
    fake = FakeTelethonClient(msgs)
    stats = asyncio.run(pipe.ingest_channel(
        owner_id=7, channel_url="https://t.me/demo", limit=None, client=fake,
    ))

    assert fake.downloads == 0  # nothing extractable -> no wasted downloads
    assert stats.processed_media == 1  # only the captioned audio is indexed (via caption)
    assert stats.skipped_media == 1    # the caption-less photo is skipped
    assert len(repo.keyword_candidates(owner_id=7, query="captioned", top_k=10, channel_url=None)) == 1
