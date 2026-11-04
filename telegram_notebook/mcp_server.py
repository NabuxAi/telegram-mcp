"""Read-only MCP server exposing a user's Telegram archive to AI clients.

Speaks JSON-RPC 2.0 over stdio (the MCP stdio transport) using only the standard
library, so it runs anywhere without extra dependencies and is easy to unit-test:
``handle_request`` is a pure dict-in/dict-out function and ``serve_stdio`` is a thin
newline-delimited loop on top of it.

The server is scoped to a single archive owner (``MCP_OWNER_ID``, default 0 — the web
dashboard archive). All tools are read-only; they never modify stored data.

Run it with:  python -m telegram_notebook.mcp_server
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

from .clustering import build_topics, label_cluster
from .config import get_settings, model_for, provider_credentials
from .db import Repository, connect
from .embeddings import EmbeddingService
from .llm import generate_text
from .logging_config import setup_logging
from .recent import recent_rows
from .search import SearchService
from .stats import format_stats
from .timeline import build_timeline

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "telegram-notebook"
SERVER_VERSION = "0.3.0"


class McpServer:
    def __init__(self, *, repository: Repository, search_service: SearchService, settings: Any, owner_id: int) -> None:
        self.repository = repository
        self.search_service = search_service
        self.settings = settings
        self.owner_id = owner_id
        self._tools = self._build_tools()

    # --- tool definitions ---

    def _build_tools(self) -> dict[str, dict[str, Any]]:
        return {
            "list_sources": {
                "description": "List the indexed Telegram sources (channels/chats and the forwarded inbox).",
                "inputSchema": {"type": "object", "properties": {}},
                "handler": self._tool_list_sources,
            },
            "list_tags": {
                "description": "List the tags in the archive and how many items each one covers.",
                "inputSchema": {"type": "object", "properties": {}},
                "handler": self._tool_list_tags,
            },
            "search_telegram_archive": {
                "description": "Keyword/semantic search over the archive. Optionally scope by source URL or tag.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for."},
                        "source": {"type": "string", "description": "Optional channel/source URL to restrict to."},
                        "tag": {"type": "string", "description": "Optional tag to restrict to."},
                        "top_k": {"type": "integer", "description": "Max results (default 5)."},
                    },
                    "required": ["query"],
                },
                "handler": self._tool_search,
            },
            "get_message": {
                "description": "Get the full stored text of a single archived item by its media_item_id.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"media_item_id": {"type": "integer"}},
                    "required": ["media_item_id"],
                },
                "handler": self._tool_get_message,
            },
            "ask_telegram_notebook": {
                "description": "Ask a question and get an AI answer grounded in the archive (optionally scoped by source/tag).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "source": {"type": "string"},
                        "tag": {"type": "string"},
                    },
                    "required": ["question"],
                },
                "handler": self._tool_ask,
            },
            "summarize_source": {
                "description": "Summarize the whole archive, a single source, or a tag.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"source": {"type": "string"}, "tag": {"type": "string"}},
                },
                "handler": self._tool_summarize,
            },
            "list_topics": {
                "description": "Cluster the archive's content into topics (offline, from stored embeddings).",
                "inputSchema": {
                    "type": "object",
                    "properties": {"source": {"type": "string"}, "tag": {"type": "string"}},
                },
                "handler": self._tool_list_topics,
            },
            "timeline": {
                "description": "Show how many items were archived per calendar period (by month, or by day).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "tag": {"type": "string"},
                        "granularity": {"type": "string", "description": "'month' (default) or 'day'."},
                    },
                },
                "handler": self._tool_timeline,
            },
            "archive_stats": {
                "description": "Overview counts for the archive: items, sources, tags, types, and date range.",
                "inputSchema": {"type": "object", "properties": {}},
                "handler": self._tool_archive_stats,
            },
            "list_recent": {
                "description": "List the most recently archived items (newest first).",
                "inputSchema": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "description": "Max items (default 10)."}},
                },
                "handler": self._tool_list_recent,
            },
        }

    # --- tool handlers (return plain text) ---

    def _tool_list_sources(self, args: dict) -> str:
        channels = self.repository.list_channels(owner_id=self.owner_id)
        if not channels:
            return "No sources indexed."
        return "\n".join(f"- {c.get('channel_title') or 'Unknown'}: {c['channel_url']}" for c in channels)

    def _tool_list_tags(self, args: dict) -> str:
        tags = self.repository.list_tags(owner_id=self.owner_id)
        if not tags:
            return "No tags yet."
        return "\n".join(f"- {t['tag']}: {t['count']}" for t in tags)

    def _tool_search(self, args: dict) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            raise ValueError("query is required")
        top_k = int(args.get("top_k") or 5)
        results = self.search_service.search(
            owner_id=self.owner_id,
            query=query,
            channel_url=args.get("source") or None,
            tag=args.get("tag") or None,
            top_k=top_k,
        )
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            source = r.channel_title or r.channel_url or "Unknown"
            snippet = (r.chunk_text or "").strip()[:400]
            line = f"[{i}] ({source}) {snippet}"
            if r.message_url:
                line += f"\n    {r.message_url}"
            lines.append(line)
        return "\n".join(lines)

    def _tool_get_message(self, args: dict) -> str:
        try:
            media_item_id = int(args["media_item_id"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("media_item_id (integer) is required") from exc
        item = self.repository.get_media_item(owner_id=self.owner_id, media_item_id=media_item_id)
        if not item:
            return "Item not found."
        source = item.get("channel_title") or item.get("channel_url") or "Unknown"
        header = f"Source: {source}"
        if item.get("message_url"):
            header += f"\nURL: {item['message_url']}"
        return f"{header}\n\n{(item.get('text') or '').strip()}"

    def _tool_ask(self, args: dict) -> str:
        question = str(args.get("question", "")).strip()
        if not question:
            raise ValueError("question is required")
        results = self.search_service.search(
            owner_id=self.owner_id,
            query=question,
            channel_url=args.get("source") or None,
            tag=args.get("tag") or None,
            top_k=5,
        )
        return self.search_service.generate_answer(
            query=question,
            results=results,
            **self._llm_kwargs(),
        )

    def _llm_kwargs(self) -> dict:
        """Generation kwargs for the configured chat provider (default: Ollama)."""
        s = self.settings
        provider = (getattr(s, "llm_provider", None) or "ollama").lower()
        api_key = None
        if provider == "gemini":
            api_key = getattr(s, "gemini_api_key", None)
        elif provider == "openai":
            api_key = getattr(s, "openai_api_key", None)
        return {
            "provider": provider,
            "model": getattr(s, "llm_model", None),
            "base_url": getattr(s, "ollama_base_url", None),
            "api_key": api_key,
            "project_id": getattr(s, "vertex_project_id", None),
            "region": getattr(s, "vertex_region", None) or "us-central1",
        }

    def _llm_available(self) -> bool:
        provider = (getattr(self.settings, "llm_provider", None) or "ollama").lower()
        if provider in ("ollama", "local"):
            return True
        if provider == "gemini":
            return bool(getattr(self.settings, "gemini_api_key", None))
        return bool(getattr(self.settings, "openai_api_key", None))

    def _tool_list_topics(self, args: dict) -> str:
        items = self.repository.chunks_with_embeddings(
            owner_id=self.owner_id, channel_url=args.get("source") or None, tag=args.get("tag") or None
        )
        if not items:
            return "No embedded content to cluster yet."
        namer = None
        if self._llm_available():
            kwargs = self._llm_kwargs()

            def namer(texts: list[str]) -> str:
                return label_cluster(
                    texts,
                    generate=lambda prompt: generate_text(prompt=prompt, **kwargs),
                )
        topics = build_topics(items, namer=namer)
        return "\n".join(f"- {t['label']} ({t['size']})" for t in topics)

    def _tool_archive_stats(self, args: dict) -> str:
        return format_stats(self.repository.archive_stats(owner_id=self.owner_id))

    def _tool_list_recent(self, args: dict) -> str:
        limit = max(1, min(50, int(args.get("limit") or 10)))
        items = self.repository.timeline_items(owner_id=self.owner_id, limit=limit)
        if not items:
            return "No content yet."
        rows = recent_rows(items, limit=limit)
        lines = []
        for r in rows:
            head = f"- ({r['source']})"
            if r["date"]:
                head += f" {r['date']}"
            if r["snippet"]:
                head += f": {r['snippet']}"
            if r["url"]:
                head += f"\n  {r['url']}"
            lines.append(head)
        return "\n".join(lines)

    def _tool_timeline(self, args: dict) -> str:
        items = self.repository.timeline_items(
            owner_id=self.owner_id, channel_url=args.get("source") or None, tag=args.get("tag") or None
        )
        if not items:
            return "No dated content yet."
        granularity = "day" if str(args.get("granularity")).lower() == "day" else "month"
        periods = build_timeline(items, granularity=granularity)
        return "\n".join(f"- {p['period']}: {p['count']}" for p in periods)

    def _tool_summarize(self, args: dict) -> str:
        source = args.get("source") or None
        tag = args.get("tag") or None
        items = self.repository.summary_items(owner_id=self.owner_id, channel_url=source, tag=tag)
        scope_label = tag or source or "the whole archive"
        return self.search_service.summarize(
            scope_label=scope_label,
            items=items,
            api_key=self.settings.gemini_api_key,
            project_id=self.settings.vertex_project_id,
            region=self.settings.vertex_region or "us-central1",
        )

    # --- JSON-RPC plumbing ---

    @staticmethod
    def _result(req_id: Any, result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

    def handle_request(self, request: dict) -> dict | None:
        method = request.get("method")
        req_id = request.get("id")

        if method == "initialize":
            return self._result(req_id, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            })

        if method in ("notifications/initialized", "initialized"):
            return None  # notification, no response

        if method == "tools/list":
            tools = [
                {"name": name, "description": meta["description"], "inputSchema": meta["inputSchema"]}
                for name, meta in self._tools.items()
            ]
            return self._result(req_id, {"tools": tools})

        if method == "tools/call":
            params = request.get("params") or {}
            name = params.get("name")
            arguments = params.get("arguments") or {}
            tool = self._tools.get(name)
            if not tool:
                return self._result(req_id, {
                    "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                    "isError": True,
                })
            try:
                text = tool["handler"](arguments)
                return self._result(req_id, {"content": [{"type": "text", "text": text}]})
            except Exception as exc:
                logger.exception("Tool %s failed", name)
                return self._result(req_id, {
                    "content": [{"type": "text", "text": f"Error: {exc}"}],
                    "isError": True,
                })

        if req_id is None:
            return None  # unknown notification
        return self._error(req_id, -32601, f"Method not found: {method}")

    def serve_stdio(self, stdin, stdout) -> None:
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON-RPC line")
                continue
            response = self.handle_request(request)
            if response is not None:
                stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                stdout.flush()


def build_mcp_server() -> McpServer:
    settings = get_settings()
    repository = Repository(connect(settings.db_path))
    repository.init()
    embed_key, embed_base = provider_credentials(settings, settings.embedding_provider)
    embeddings = EmbeddingService(
        provider=settings.embedding_provider,
        api_key=embed_key,
        base_url=embed_base,
        model=model_for(settings, settings.embedding_provider, "embedding"),
    )
    search_service = SearchService(repository, embeddings)
    owner_id = int(os.environ.get("MCP_OWNER_ID") or 0)
    return McpServer(repository=repository, search_service=search_service, settings=settings, owner_id=owner_id)


def main() -> None:
    setup_logging()
    server = build_mcp_server()
    logger.info("MCP server ready (owner_id=%s)", server.owner_id)
    server.serve_stdio(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
