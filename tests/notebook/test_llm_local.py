"""Local-first provider routing: Ollama chat + embeddings, plus fallbacks."""

from telegram_notebook import llm
from telegram_notebook.embeddings import EmbeddingService
from telegram_notebook.transcription import TranscriptionService


def test_generate_text_routes_to_ollama(monkeypatch):
    captured = {}

    def fake_post(url, payload, *, timeout):
        captured["url"] = url
        captured["payload"] = payload
        return {"response": "  local answer  "}

    monkeypatch.setattr(llm, "_post_json", fake_post)
    out = llm.generate_text(provider="ollama", model="llama3.1", prompt="hi", base_url="http://host:11434")
    assert out == "local answer"  # trimmed
    assert captured["url"] == "http://host:11434/api/generate"
    assert captured["payload"]["model"] == "llama3.1"
    assert captured["payload"]["stream"] is False


def test_generate_text_defaults_to_ollama_base(monkeypatch):
    seen = {}

    def fake_post(url, payload, *, timeout):
        seen["url"] = url
        return {"response": "x"}

    monkeypatch.setattr(llm, "_post_json", fake_post)
    llm.generate_text(provider=None, prompt="hi")  # provider None -> ollama
    assert seen["url"].startswith(llm.DEFAULT_OLLAMA_BASE_URL)


def test_generate_text_routes_to_gemini(monkeypatch):
    calls = {}

    def fake_gemini(*, api_key, model, prompt, project_id, region):
        calls["model"] = model
        calls["prompt"] = prompt
        return "gemini says hi"

    monkeypatch.setattr("telegram_notebook.provider_http.gemini_generate_content", fake_gemini)
    out = llm.generate_text(provider="gemini", model="gemini-2.5-flash-lite", prompt="q", api_key="k")
    assert out == "gemini says hi"
    assert calls["prompt"] == "q"


def test_ollama_embeddings(monkeypatch):
    monkeypatch.setattr(llm, "_post_json", lambda url, payload, *, timeout: {"embedding": [0.1, 0.2, 0.3]})
    svc = EmbeddingService(provider="ollama", api_key=None, model="nomic-embed-text", base_url="http://h:11434")
    assert svc.enabled is True  # no API key needed for local
    assert svc.embed("hello") == [0.1, 0.2, 0.3]


def test_openai_embeddings_still_need_key():
    svc = EmbeddingService(provider="openai", api_key=None, model="text-embedding-3-small")
    assert svc.enabled is False  # cloud still requires a key


def test_local_transcription_enabled_without_key():
    svc = TranscriptionService(provider="local", api_key=None, model="base")
    assert svc.enabled is True


def test_cloud_transcription_requires_key():
    svc = TranscriptionService(provider="openai", api_key=None, model="whisper-1")
    assert svc.enabled is False
