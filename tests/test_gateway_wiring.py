"""Every AI service must be reachable at the gateway, not just keyed for it.

NabuGate speaks the OpenAI wire protocol, so the only thing that distinguishes
"call the gateway" from "call OpenAI" is base_url. A service constructed with
the gateway's token but no base_url sends that token to api.openai.com — it
fails, and the failure says "invalid api key", which points at the wrong thing.

Transcription had exactly that gap while embeddings did not, so this asserts on
the shape rather than on one example.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Services that talk to a provider over the OpenAI protocol.
WIRE_SERVICES = ("EmbeddingService", "TranscriptionService")

#: Files that construct them for real (tests and helpers excluded).
SOURCES = [
    ROOT / "app" / "notebook.py",
    ROOT / "telegram_notebook" / "bot.py",
    ROOT / "telegram_notebook" / "main.py",
]


def _constructions(path: pathlib.Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in WIRE_SERVICES:
            yield node


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_every_wire_service_is_given_a_base_url(path):
    found = list(_constructions(path))
    assert found, f"no service constructions in {path.name} — the test would pass vacuously"
    for call in found:
        kw = {k.arg for k in call.keywords}
        assert "base_url" in kw, (
            f"{path.name}:{call.lineno} builds {call.func.id} without base_url — "
            "with a gateway token that request goes to the vendor, not the gateway"
        )


def _enclosing_function(tree, call):
    """The function a call sits inside, so a model resolved a few lines above counts."""
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= call.lineno <= (node.end_lineno or node.lineno):
                if best is None or node.lineno > best.lineno:
                    best = node
    return best


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_every_wire_service_resolves_its_model_for_the_provider(path):
    """A model name chosen for Ollama or Gemini is not an alias the gateway knows.

    Either passed inline, or resolved in the same function — the per-user paths
    legitimately keep a caller's preference for the vendors and only override it
    for the gateway, which reads as a conditional a few lines up.
    """
    tree = ast.parse(path.read_text())
    for call in _constructions(path):
        model = next((k.value for k in call.keywords if k.arg == "model"), None)
        assert model is not None, f"{path.name}:{call.lineno} passes no model"
        inline = ast.unparse(model)
        if "model_for" in inline:
            continue
        fn = _enclosing_function(tree, call)
        scope = ast.unparse(fn) if fn else ""
        assert "model_for" in scope, (
            f"{path.name}:{call.lineno} passes {inline!r} with no model_for() anywhere in "
            f"{(fn.name if fn else '<module>')}; a provider-specific name would reach the "
            "gateway as an unknown alias"
        )


def test_the_openai_client_is_always_built_with_base_url():
    """The constructor itself, not just its callers."""
    for name in ("embeddings.py", "transcription.py", "llm.py"):
        src = (ROOT / "telegram_notebook" / name).read_text()
        for line in src.splitlines():
            if "OpenAI(" in line and "import" not in line:
                assert "base_url" in line, f"{name}: {line.strip()!r} pins the vendor"


def test_transcription_has_a_gateway_alias_of_its_own():
    from telegram_notebook import config

    assert config.NABUGATE_DEFAULT_TRANSCRIBE_MODEL.startswith("nabu-")
    settings = config.get_settings()
    object.__setattr__(settings, "transcription_model", "gemini-2.5-flash-lite")
    # A Gemini model name must not survive into a gateway call.
    assert config.model_for(settings, "nabugate", "transcription") == \
        config.NABUGATE_DEFAULT_TRANSCRIBE_MODEL
    # A real alias must.
    object.__setattr__(settings, "transcription_model", "nabu-transcribe")
    assert config.model_for(settings, "nabugate", "transcription") == "nabu-transcribe"
    # And a non-gateway provider keeps whatever it was given.
    object.__setattr__(settings, "transcription_model", "gemini-2.5-flash-lite")
    assert config.model_for(settings, "gemini", "transcription") == "gemini-2.5-flash-lite"
