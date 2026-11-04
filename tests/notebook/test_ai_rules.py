from types import SimpleNamespace

from telegram_notebook import bot as bot_module
from telegram_notebook.bot import NotebookBot
from telegram_notebook.db import Repository
from telegram_notebook.rules import (
    build_classify_prompt,
    classify_ai_tags,
    match_tags,
    parse_classified_tags,
)


class FakeApi:
    def __init__(self):
        self.sent = []

    def send_message(self, chat_id=None, text=None, **kwargs):
        self.sent.append((chat_id, text))

# --- pure matching helpers ---

def test_match_tags_ignores_ai_rules():
    rules = [
        {"keyword": "villa", "tag": "Real Estate", "kind": "keyword"},
        {"keyword": "anything about money", "tag": "Finance", "kind": "ai"},
    ]
    # keyword rule matches; ai rule is not evaluated here even if its text appears
    assert match_tags("a villa with money", rules) == {"Real Estate"}


def test_build_classify_prompt_lists_criteria_and_truncates():
    rules = [{"keyword": "is about real estate", "tag": "RE", "kind": "ai"}]
    prompt = build_classify_prompt("x" * 5000, rules, max_chars=100)
    assert "RE: is about real estate" in prompt
    assert "x" * 100 in prompt and "x" * 101 not in prompt  # truncated


def test_parse_classified_tags_keeps_only_valid():
    rules = [{"tag": "RE", "kind": "ai"}, {"tag": "Finance", "kind": "ai"}]
    assert parse_classified_tags("RE, Finance", rules) == {"RE", "Finance"}
    assert parse_classified_tags("re\nfinance", rules) == {"RE", "Finance"}  # case-insensitive, newlines
    assert parse_classified_tags("NONE", rules) == set()
    assert parse_classified_tags("Sports, RE", rules) == {"RE"}  # unknown tag dropped


def test_classify_ai_tags_uses_injected_generate():
    rules = [{"keyword": "about claude", "tag": "AI", "kind": "ai"}]
    calls = []

    def fake_generate(prompt: str) -> str:
        calls.append(prompt)
        return "AI"

    assert classify_ai_tags("a note about claude", rules, generate=fake_generate) == {"AI"}
    assert len(calls) == 1
    # short-circuits without calling the model
    calls.clear()
    assert classify_ai_tags("", rules, generate=fake_generate) == set()
    assert classify_ai_tags("text", [], generate=fake_generate) == set()
    assert calls == []


# --- repository: rule kind persistence + migration ---

def test_add_rule_stores_kind(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    repo.add_rule(owner_id=1, keyword="villa", tag="RE")  # default keyword
    repo.add_rule(owner_id=1, keyword="is about money", tag="Finance", kind="ai")
    rules = repo.list_rules(owner_id=1)
    by_tag = {r["tag"]: r for r in rules}
    assert by_tag["RE"]["kind"] == "keyword"
    assert by_tag["Finance"]["kind"] == "ai"


# --- bot /rule apply: keyword + AI rules combined ---

def _seed_item(repo: Repository, *, owner_id: int, text: str) -> int:
    cid = repo.upsert_channel(owner_id=owner_id, telegram_id=1, channel_url="https://t.me/c", title="C", username=None)
    mid = repo.create_or_get_message(channel_id=cid, telegram_message_id=hash(text) % 9999, message_date="2026-06-01T00:00:00", message_url="u", caption=None)
    media_id = repo.create_or_get_media(message_id=mid, file_name="t", file_path="", mime_type="text/plain", media_kind="text", duration_seconds=None, file_size_bytes=1)
    repo.mark_media_transcribed(media_item_id=media_id, transcript_text=text)
    return media_id


def test_retag_all_applies_keyword_and_ai_rules(tmp_path, monkeypatch):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    repo.add_rule(owner_id=1, keyword="villa", tag="RE")
    repo.add_rule(owner_id=1, keyword="is about an AI tool", tag="AI", kind="ai")
    _seed_item(repo, owner_id=1, text="a villa review built with claude")

    captured = {}

    def fake_generate(*, api_key=None, prompt=None, project_id=None, region=None):
        captured["prompt"] = prompt
        return "AI"  # the model decides the AI rule matches

    monkeypatch.setattr(bot_module, "gemini_generate_content", fake_generate)

    api = FakeApi()
    fake_bot = SimpleNamespace(services=SimpleNamespace(
        repository=repo,
        api=api,
        settings=SimpleNamespace(gemini_api_key="key", vertex_project_id=None, vertex_region="us-central1"),
    ))
    NotebookBot._retag_all(fake_bot, 10, 1)

    tags = {t["tag"] for t in repo.list_tags(owner_id=1)}
    assert tags == {"RE", "AI"}  # keyword + AI rule both applied
    assert "is about an AI tool" in captured["prompt"]
    assert "Re-tagged 1" in api.sent[-1][1]


def test_retag_all_skips_ai_without_api_key(tmp_path, monkeypatch):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    repo.add_rule(owner_id=1, keyword="is about money", tag="Finance", kind="ai")
    _seed_item(repo, owner_id=1, text="quarterly revenue and profit")

    def boom(*args, **kwargs):
        raise AssertionError("LLM must not be called without an API key")

    monkeypatch.setattr(bot_module, "gemini_generate_content", boom)

    api = FakeApi()
    fake_bot = SimpleNamespace(services=SimpleNamespace(
        repository=repo,
        api=api,
        settings=SimpleNamespace(gemini_api_key=None, vertex_project_id=None, vertex_region="us-central1"),
    ))
    NotebookBot._retag_all(fake_bot, 10, 1)

    assert repo.list_tags(owner_id=1) == []  # nothing tagged
    assert "skipped" in api.sent[-1][1]
