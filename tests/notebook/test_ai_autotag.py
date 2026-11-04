import asyncio
from types import SimpleNamespace

from telegram_notebook.bot import NotebookBot
from telegram_notebook.db import Repository
from telegram_notebook.embeddings import EmbeddingService
from telegram_notebook.pipeline import IngestionPipeline


class _Probe(NotebookBot):
    def __init__(self):
        pass


class FakeApi:
    def __init__(self):
        self.messages = []

    def send_message(self, chat_id=None, text=None, **kwargs):
        self.messages.append((chat_id, text))


def _pipeline(repo, *, ai_classifier=None):
    return IngestionPipeline(
        settings=SimpleNamespace(chunk_size=900, chunk_overlap=120),
        repository=repo,
        transcription=None,
        embeddings=EmbeddingService(provider="openai", api_key=None, model="x"),
        ai_classifier=ai_classifier,
    )


def test_pipeline_applies_ai_rules_only_with_classifier(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    repo.add_rule(owner_id=1, keyword="villa", tag="RE")  # keyword
    repo.add_rule(owner_id=1, keyword="about an AI tool", tag="AI", kind="ai")

    # Without a classifier: only keyword rules apply.
    asyncio.run(_pipeline(repo).ingest_forwarded_message(
        owner_id=1, source_label="s", text="a villa built with claude", forward_key=1,
    ))
    assert {t["tag"] for t in repo.list_tags(owner_id=1)} == {"RE"}

    # With a classifier: AI rule also applies.
    calls = []

    def classifier(prompt):
        calls.append(prompt)
        return "AI"

    asyncio.run(_pipeline(repo, ai_classifier=classifier).ingest_forwarded_message(
        owner_id=1, source_label="s", text="another villa with an ai tool", forward_key=2,
    ))
    assert {t["tag"] for t in repo.list_tags(owner_id=1)} == {"RE", "AI"}
    assert len(calls) == 1


def test_pipeline_ai_classifier_error_is_swallowed(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    repo.add_rule(owner_id=1, keyword="x", tag="AI", kind="ai")

    def boom(prompt):
        raise RuntimeError("llm down")

    # Keyword match still works; AI failure does not break ingest.
    repo.add_rule(owner_id=1, keyword="villa", tag="RE")
    asyncio.run(_pipeline(repo, ai_classifier=boom).ingest_forwarded_message(
        owner_id=1, source_label="s", text="a villa", forward_key=1,
    ))
    assert {t["tag"] for t in repo.list_tags(owner_id=1)} == {"RE"}


def test_set_ai_autotag_persists(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    repo.upsert_bot_user(bot_user_id=1, chat_id=10, username=None, first_name=None)
    assert repo.get_bot_user(bot_user_id=1)["ai_autotag"] == 0  # default off
    repo.set_ai_autotag(bot_user_id=1, enabled=True)
    assert repo.get_bot_user(bot_user_id=1)["ai_autotag"] == 1
    repo.set_ai_autotag(bot_user_id=1, enabled=False)
    assert repo.get_bot_user(bot_user_id=1)["ai_autotag"] == 0


def test_ai_classifier_for_user_gating():
    p = _Probe()
    p.services = SimpleNamespace(settings=SimpleNamespace(gemini_api_key="key", vertex_project_id=None, vertex_region="us-central1"))
    assert p._ai_classifier_for_user({"ai_autotag": 0, "gemini_api_key": "k"}) is None  # opted out
    assert p._ai_classifier_for_user(None) is None
    # opted in + key present -> a callable
    assert callable(p._ai_classifier_for_user({"ai_autotag": 1, "gemini_api_key": None}))
    # opted in but no key anywhere -> None
    p.services.settings.gemini_api_key = None
    assert p._ai_classifier_for_user({"ai_autotag": 1, "gemini_api_key": None}) is None


def test_handle_airules(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    repo.upsert_bot_user(bot_user_id=1, chat_id=10, username=None, first_name=None)
    p = _Probe()
    api = FakeApi()
    p.services = SimpleNamespace(repository=repo, api=api, settings=SimpleNamespace(gemini_api_key="key"))

    p._handle_airules(10, 1, "on")
    assert "ON" in api.messages[-1][1]
    assert repo.get_bot_user(bot_user_id=1)["ai_autotag"] == 1

    p._handle_airules(10, 1, "")  # status
    assert "ON" in api.messages[-1][1]

    p._handle_airules(10, 1, "off")
    assert "OFF" in api.messages[-1][1]
    assert repo.get_bot_user(bot_user_id=1)["ai_autotag"] == 0
