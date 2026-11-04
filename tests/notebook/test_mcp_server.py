import asyncio
import io
import json
from types import SimpleNamespace

from telegram_notebook.db import Repository
from telegram_notebook.embeddings import EmbeddingService
from telegram_notebook.mcp_server import PROTOCOL_VERSION, McpServer
from telegram_notebook.pipeline import IngestionPipeline
from telegram_notebook.search import SearchService


def _server(tmp_path, owner_id=1):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    embeddings = EmbeddingService(provider="openai", api_key=None, model="x")  # disabled, offline
    search = SearchService(repo, embeddings)
    settings = SimpleNamespace(gemini_api_key=None, vertex_project_id=None, vertex_region="us-central1")
    server = McpServer(repository=repo, search_service=search, settings=settings, owner_id=owner_id)
    return server, repo


def _seed(repo: Repository, owner_id=1):
    pipe = IngestionPipeline(
        settings=SimpleNamespace(chunk_size=900, chunk_overlap=120),
        repository=repo,
        transcription=None,
        embeddings=EmbeddingService(provider="openai", api_key=None, model="x"),
    )
    asyncio.run(pipe.ingest_forwarded_message(owner_id=owner_id, source_label="AI News", text="a note about townhouses", forward_key=1))


def _call(server, name, arguments=None):
    resp = server.handle_request({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": arguments or {}},
    })
    return resp["result"]


def test_initialize_and_tools_list(tmp_path):
    server, _ = _server(tmp_path)
    init = server.handle_request({"jsonrpc": "2.0", "id": 0, "method": "initialize"})
    assert init["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert init["result"]["serverInfo"]["name"] == "telegram-notebook"
    assert "tools" in init["result"]["capabilities"]

    listed = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = {t["name"] for t in listed["result"]["tools"]}
    assert {"list_sources", "list_tags", "search_telegram_archive", "get_message",
            "ask_telegram_notebook", "summarize_source"} <= names
    # every tool advertises an input schema
    assert all("inputSchema" in t for t in listed["result"]["tools"])


def test_initialized_notification_has_no_response(tmp_path):
    server, _ = _server(tmp_path)
    assert server.handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_unknown_method_returns_error(tmp_path):
    server, _ = _server(tmp_path)
    resp = server.handle_request({"jsonrpc": "2.0", "id": 9, "method": "no/such"})
    assert resp["error"]["code"] == -32601


def test_list_sources_and_search(tmp_path):
    server, repo = _server(tmp_path)
    _seed(repo)
    sources = _call(server, "list_sources")
    assert "Forwarded Inbox" in sources["content"][0]["text"]

    found = _call(server, "search_telegram_archive", {"query": "townhouses"})
    assert "townhouses" in found["content"][0]["text"]

    nothing = _call(server, "search_telegram_archive", {"query": "nonexistentword"})
    assert "No results" in nothing["content"][0]["text"]


def test_get_message_and_unknown_tool(tmp_path):
    server, repo = _server(tmp_path)
    _seed(repo)
    # media_item_id 1 is the first stored forwarded item
    item = _call(server, "get_message", {"media_item_id": 1})
    assert "townhouses" in item["content"][0]["text"]

    missing = _call(server, "get_message", {"media_item_id": 999})
    assert "not found" in missing["content"][0]["text"].lower()

    bad = _call(server, "does_not_exist", {})
    assert bad["isError"] is True


def test_owner_isolation_in_mcp(tmp_path):
    server, repo = _server(tmp_path, owner_id=2)  # server scoped to a different owner
    _seed(repo, owner_id=1)
    assert "No sources" in _call(server, "list_sources")["content"][0]["text"]
    assert "No results" in _call(server, "search_telegram_archive", {"query": "townhouses"})["content"][0]["text"]


def test_serve_stdio_roundtrip(tmp_path):
    server, _ = _server(tmp_path)
    stdin = io.StringIO(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n"
        + json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
    )
    stdout = io.StringIO()
    server.serve_stdio(stdin, stdout)
    out_lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
    # one response for tools/list, none for the notification
    assert len(out_lines) == 1
    assert json.loads(out_lines[0])["id"] == 1
