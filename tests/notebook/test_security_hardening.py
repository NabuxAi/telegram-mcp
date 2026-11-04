"""Regression tests for the three defects found in the 2026-09-03 audit."""

from __future__ import annotations

from pathlib import Path

import pytest

from telegram_notebook import config
from telegram_notebook.bot_api import safe_filename
from telegram_notebook.clustering import cluster_embeddings

# --- bot_api: a Telegram-supplied file name must not escape the temp dir ---

@pytest.mark.parametrize(
    "supplied",
    [
        "../../../../etc/passwd",
        "/etc/cron.d/evil",
        "..",
        ".",
        "",
        None,
        "../.env",
        "sub/dir/payload.txt",
    ],
)
def test_safe_filename_yields_one_harmless_segment(supplied):
    name = safe_filename(supplied)
    assert name, "must never return an empty name"
    assert Path(name).name == name, "must be a single path segment"
    assert not name.startswith("."), "must not produce a dotfile or a traversal"
    # The decisive property: joining it can never leave the directory.
    base = Path("/tmp/dl").resolve()
    assert (base / name).resolve().parent == base


def test_safe_filename_keeps_ordinary_names():
    assert safe_filename("archive.zip") == "archive.zip"
    assert safe_filename("my-export_2026.json") == "my-export_2026.json"


# --- config: a newline in a settings value must not define a second variable ---

def test_upsert_env_rejects_newline_injection(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("WEB_API_TOKEN=legit\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", env)

    with pytest.raises(ValueError):
        config.upsert_env_values({"GEMINI_API_KEY": "x\nWEB_API_TOKEN=attacker"})

    # The file must be untouched — the whole point is that nothing is written.
    assert env.read_text(encoding="utf-8") == "WEB_API_TOKEN=legit\n"


def test_upsert_env_still_writes_ordinary_values(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("A=1\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", env)

    config.upsert_env_values({"A": "2", "B": "three"})
    body = env.read_text(encoding="utf-8")
    assert "A=2" in body
    assert "B=three" in body


def test_upsert_env_blanks_none(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("TOKEN=secret\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", env)

    config.upsert_env_values({"TOKEN": None})
    assert "TOKEN=\n" in env.read_text(encoding="utf-8")


# --- clustering: mixed embedding dimensions must not abort the run ---

def test_cluster_embeddings_survives_mixed_dimensions():
    """An archive re-embedded with a different model holds two vector sizes."""
    items = [
        {"embedding": [1.0, 0.0, 0.0, 0.0]},
        {"embedding": [0.99, 0.01, 0.0, 0.0]},
        {"embedding": [0.0, 1.0, 0.0, 0.0]},
        {"embedding": [1.0, 0.0]},  # a 2-dim straggler from the old model
    ]
    clusters = cluster_embeddings(items, threshold=0.99, max_clusters=3)
    assert clusters, "must return clusters rather than raising"
    total = sum(len(c) for c in clusters)
    assert total == len(items), "every item must land in exactly one cluster"


# --- NabuGate: a stored index must never receive a second embedding space ---

def test_embedding_service_refuses_a_wrong_width_from_the_gateway():
    """gemini-embedding-001 returns 3072 unless asked for 1536.

    Storing that vector would not fail — it would sit in the same column and
    return a plausible cosine that means nothing. Refusing is the whole point.
    """
    from telegram_notebook import embeddings as emb

    class _Resp:
        def __init__(self, vec):
            self.data = [type("D", (), {"embedding": vec})()]

    class _Client:
        def __init__(self, vec):
            self.seen = {}
            self.embeddings = type(
                "E", (), {"create": lambda _s, **kw: (self.seen.update(kw), _Resp(vec))[1]}
            )()

    svc = emb.EmbeddingService(
        provider="nabugate", api_key="k", model="notebook-embed",
        base_url="https://gate.nabuxai.com/v1",
    )
    svc.client = _Client([0.0] * 3072)

    with pytest.raises(emb.EmbeddingDimensionError):
        svc.embed("سلام")

    # And it must have asked for the right width in the first place.
    assert svc.client.seen.get("dimensions") == emb.NABUGATE_EMBED_DIM


def test_embedding_service_accepts_the_configured_width():
    from telegram_notebook import embeddings as emb

    class _Resp:
        def __init__(self, vec):
            self.data = [type("D", (), {"embedding": vec})()]

    class _Client:
        def __init__(self, vec):
            self.embeddings = type("E", (), {"create": lambda _s, **kw: _Resp(vec)})()

    svc = emb.EmbeddingService(
        provider="nabugate", api_key="k", model="notebook-embed", base_url="x",
    )
    svc.client = _Client([0.1] * emb.NABUGATE_EMBED_DIM)
    assert len(svc.embed("سلام")) == emb.NABUGATE_EMBED_DIM


def test_empty_completion_is_a_failure_not_an_answer(monkeypatch):
    """The gateway already walked its whole chain to produce this nothing."""
    from telegram_notebook import llm

    monkeypatch.setattr(llm, "openai_generate", lambda **kw: "   ")
    with pytest.raises(llm.EmptyCompletionError):
        llm.generate_text(provider="nabugate", model="nabu-fast", prompt="hi", api_key="k")


def test_gateway_is_the_default_provider_once_its_key_is_set(monkeypatch):
    from telegram_notebook import config

    monkeypatch.setenv("NABUGATE_API_KEY", "tok")
    assert config._default_provider() == "nabugate"
    monkeypatch.setenv("NABUGATE_API_KEY", "")
    assert config._default_provider() == "ollama"


def test_gateway_model_defaults_do_not_leak_an_ollama_model_name():
    """EMBEDDING_MODEL=nomic-embed-text must not be sent to the gateway."""
    from telegram_notebook import config

    settings = config.get_settings()
    object.__setattr__(settings, "embedding_model", "nomic-embed-text")
    assert config.model_for(settings, "nabugate", "embedding") == config.NABUGATE_DEFAULT_EMBED_MODEL
    object.__setattr__(settings, "embedding_model", "notebook-embed")
    assert config.model_for(settings, "nabugate", "embedding") == "notebook-embed"
