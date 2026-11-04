from pathlib import Path
from types import SimpleNamespace

from telegram_notebook.bot import NotebookBot
from telegram_notebook.db import Repository
from telegram_notebook.podcast import PodcastUnavailable, build_podcast_source


class _Probe(NotebookBot):
    def __init__(self):
        pass


class FakeApi:
    def __init__(self):
        self.messages = []
        self.audios = []

    def send_message(self, chat_id=None, text=None, **kwargs):
        self.messages.append((chat_id, text))

    def send_audio(self, *, chat_id, audio_path, caption=None, title=None):
        self.audios.append((chat_id, Path(audio_path).name, caption))


def _seed(repo, *, owner_id, text):
    cid = repo.upsert_channel(owner_id=owner_id, telegram_id=1, channel_url="https://t.me/c", title="C", username=None)
    mid = repo.create_or_get_message(channel_id=cid, telegram_message_id=hash(text) % 9999, message_date="2026-06-01T00:00:00", message_url="https://t.me/c/1", caption=None)
    media_id = repo.create_or_get_media(message_id=mid, file_name="t", file_path="", mime_type="text/plain", media_kind="text", duration_seconds=None, file_size_bytes=1)
    repo.mark_media_transcribed(media_item_id=media_id, transcript_text=text)


def _settings(**over):
    base = dict(gemini_api_key="key", openai_api_key=None, podcast_tts_model="edge", podcast_llm_model=None)
    base.update(over)
    return SimpleNamespace(**base)


# --- pure source builder ---

def test_build_podcast_source_structure():
    items = [
        {"text": "first note", "channel_title": "AI News"},
        {"text": "second", "channel_url": "https://t.me/b"},
    ]
    src = build_podcast_source("your whole archive", items)
    assert "Telegram archive — your whole archive" in src
    assert "AI News: first note" in src
    assert "https://t.me/b: second" in src


def test_build_podcast_source_caps_and_truncates():
    items = [{"text": "x" * 5000, "channel_title": "C"} for _ in range(50)]
    src = build_podcast_source("s", items, max_items=3, per_item_chars=100)
    assert src.count("C: ") == 3          # capped to 3 items
    assert "x" * 100 in src and "x" * 101 not in src  # each truncated to 100 chars


def test_build_podcast_source_skips_empty_text():
    items = [{"text": "", "channel_title": "C"}, {"text": "real", "channel_title": "D"}]
    src = build_podcast_source("s", items)
    assert "D: real" in src
    assert "C:" not in src


# --- /podcast handler orchestration ---

def test_handle_podcast_sends_audio(tmp_path, monkeypatch):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    _seed(repo, owner_id=1, text="content about AI tools")
    api = FakeApi()
    probe = _Probe()
    probe.services = SimpleNamespace(repository=repo, api=api, settings=_settings())

    audio = tmp_path / "podcast.mp3"
    audio.write_bytes(b"fake-audio")
    captured = {}

    def fake_synth(*, text, gemini_api_key, openai_api_key, tts_model, llm_model_name):
        captured.update(text=text, gemini=gemini_api_key, tts=tts_model)
        return str(audio)

    monkeypatch.setattr("telegram_notebook.bot.synthesize_podcast", fake_synth)
    probe._handle_podcast(10, 1, "")

    assert len(api.audios) == 1
    chat_id, name, caption = api.audios[0]
    assert chat_id == 10 and name == "podcast.mp3"
    assert "Audio Overview" in caption
    assert "content about AI tools" in captured["text"]
    assert captured["gemini"] == "key" and captured["tts"] == "edge"
    assert not audio.exists()  # cleaned up after sending


def test_handle_podcast_reports_when_unavailable(tmp_path, monkeypatch):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    _seed(repo, owner_id=1, text="something")
    api = FakeApi()
    probe = _Probe()
    probe.services = SimpleNamespace(repository=repo, api=api, settings=_settings())

    def fake_synth(**_kwargs):
        raise PodcastUnavailable("install the podcast extra")

    monkeypatch.setattr("telegram_notebook.bot.synthesize_podcast", fake_synth)
    probe._handle_podcast(10, 1, "")

    assert api.audios == []
    assert "install the podcast extra" in api.messages[-1][1]


def test_handle_podcast_requires_llm_key(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    _seed(repo, owner_id=1, text="something")
    api = FakeApi()
    probe = _Probe()
    probe.services = SimpleNamespace(repository=repo, api=api, settings=_settings(gemini_api_key=None, openai_api_key=None))

    probe._handle_podcast(10, 1, "")
    assert api.audios == []
    assert "needs an LLM key" in api.messages[-1][1]


def test_handle_podcast_empty_archive(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    api = FakeApi()
    probe = _Probe()
    probe.services = SimpleNamespace(repository=repo, api=api, settings=_settings())

    probe._handle_podcast(10, 1, "")
    assert api.audios == []
    assert "Nothing to turn into a podcast" in api.messages[-1][1]
