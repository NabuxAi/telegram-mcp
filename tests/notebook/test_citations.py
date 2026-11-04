import html
from types import SimpleNamespace

from telegram_notebook import citations
from telegram_notebook.bot import NotebookBot
from telegram_notebook.db import Repository
from telegram_notebook.models import SearchResult

# --- cited_indices ---

def test_cited_indices_basic():
    assert citations.cited_indices("a [1] b [2] c [1]", 3) == [1, 2]


def test_cited_indices_grouped_and_commas():
    assert citations.cited_indices("x [1, 3] y", 3) == [1, 3]
    assert citations.cited_indices("x [1، 2] z", 3) == [1, 2]
    assert citations.cited_indices("a [1][3] b", 3) == [1, 3]


def test_cited_indices_out_of_range_filtered():
    assert citations.cited_indices("a [5] b [2]", 3) == [2]


def test_cited_indices_none():
    assert citations.cited_indices("no citations here", 3) == []
    assert citations.cited_indices("", 3) == []


# --- source_label ---

def test_source_label_prefers_title_then_url():
    assert citations.source_label(SearchResult(score=1.0, channel_title="A", channel_url="u")) == "A"
    assert citations.source_label(SearchResult(score=1.0, channel_title=None, channel_url="u")) == "u"
    assert citations.source_label({"channel_title": None, "channel_url": None}) == "Unknown source"


# --- format_sources_block ---

def _results():
    return [
        SearchResult(score=0.9, channel_title="Chan A", channel_url="https://t.me/a", message_url="https://t.me/a/1"),
        SearchResult(score=0.8, channel_title="Chan B", channel_url="https://t.me/b", message_url="https://t.me/b/2"),
        SearchResult(score=0.7, channel_title="Chan C", channel_url="https://t.me/c", message_url=None),
    ]


def test_format_sources_block_only_cited_preserves_numbers():
    block = citations.format_sources_block(_results(), [1, 3])
    assert block == "[1] Chan A — https://t.me/a/1\n[3] Chan C"
    assert "Chan B" not in block


def test_format_sources_block_fallback_all_with_limit():
    block = citations.format_sources_block(_results(), None, limit=2)
    assert block.startswith("[1] Chan A") and "[2] Chan B" in block and "Chan C" not in block


def test_format_sources_block_escapes():
    rows = [SearchResult(score=1.0, channel_title="A & B <x>", channel_url="u", message_url="m")]
    block = citations.format_sources_block(rows, [1], escape=html.escape)
    assert "A &amp; B &lt;x&gt;" in block


# --- bot _ask_brain renders numbered, citation-matched sources ---

class _Probe(NotebookBot):
    def __init__(self):
        pass


class FakeApi:
    def __init__(self):
        self.messages = []

    def send_message(self, chat_id=None, text=None, **kwargs):
        self.messages.append((chat_id, text))


class FakeSearch:
    def __init__(self, results, answer):
        self._results = results
        self._answer = answer

    def search(self, **kwargs):
        return self._results

    def generate_answer(self, **kwargs):
        return self._answer


def _probe_with_user(tmp_path, api):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    repo.upsert_bot_user(bot_user_id=1, chat_id=10, username="u", first_name="U")
    probe = _Probe()
    probe.services = SimpleNamespace(
        repository=repo, api=api,
        settings=SimpleNamespace(gemini_api_key=None, vertex_project_id=None, vertex_region=None),
    )
    probe._vertex_search_config = lambda user: None
    return probe


def test_ask_brain_lists_only_cited_sources(tmp_path):
    api = FakeApi()
    probe = _probe_with_user(tmp_path, api)
    probe._search_service_for_user = lambda user: FakeSearch(_results(), "Grounded answer [1].")
    probe._ask_brain(10, 1, "q", None)
    sources_msg = api.messages[-1][1]
    assert "Sources:" in sources_msg
    assert "[1] Chan A" in sources_msg
    assert "Chan B" not in sources_msg and "Chan C" not in sources_msg


def test_ask_brain_falls_back_when_no_citation(tmp_path):
    api = FakeApi()
    probe = _probe_with_user(tmp_path, api)
    probe._search_service_for_user = lambda user: FakeSearch(_results(), "Answer without any citation.")
    probe._ask_brain(10, 1, "q", None)
    sources_msg = api.messages[-1][1]
    assert "[1] Chan A" in sources_msg and "[2] Chan B" in sources_msg and "[3] Chan C" in sources_msg
