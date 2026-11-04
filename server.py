#!/usr/bin/env python3
"""telegram_mcp — multi-user Telegram MCP server with a NotebookLM-style brain.

One HTTPS endpoint (`PUBLIC_BASE_URL/mcp`) that any MCP client (claude.ai,
Claude Desktop, Claude Code, Cursor, ...) connects to via OAuth 2.1. The
authorization step is a plain web page where the person logs into *their own*
Telegram account (phone -> code -> optional 2FA). From then on every tool call
carries a Bearer token that maps to that person's encrypted Telegram session.

Tool families
  telegram_*  - act on the user's Telegram: chats, messages, files, groups,
                channels, contacts, profile, backups
  notebook_*  - the brain: import chats into a private searchable archive,
                semantic search, grounded Q&A with citations, summaries, topics

Run:  python server.py     (see .env.example for configuration)
"""
import asyncio
import contextlib
import logging
import re
import sys

from app import config


def build_app():
    from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
    from mcp.server.fastmcp import FastMCP
    from pydantic import AnyHttpUrl

    from app import notebook, tools, web
    from app.db import get_db
    from app.oauth import SCOPE, get_provider
    from app.tg import ClientPool, LoginManager

    db = get_db()
    provider = get_provider()
    mcp = FastMCP(
        config.APP_NAME,
        host=config.MCP_HOST,
        port=config.MCP_PORT,
        streamable_http_path=config.MCP_PATH,
        stateless_http=True,
        json_response=True,
        auth=AuthSettings(
            issuer_url=AnyHttpUrl(config.PUBLIC_BASE_URL),
            resource_server_url=AnyHttpUrl(config.PUBLIC_BASE_URL + config.MCP_PATH),
            client_registration_options=ClientRegistrationOptions(
                enabled=True, valid_scopes=[SCOPE], default_scopes=[SCOPE]),
            revocation_options=RevocationOptions(enabled=True),
        ),
        auth_server_provider=provider,
    )

    pool = ClientPool(db)
    logins = LoginManager(pool, db)
    nb = notebook.Notebook(db, pool)
    logins.on_login = nb.sync_login

    tools.register(mcp, pool)
    notebook.register_tools(mcp, nb, pool)
    web.register(mcp, provider, logins, pool, on_disconnect=nb.sync_logout, notebook_stats=nb.stats_text)

    app = mcp.streamable_http_app()

    inner = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(a):
        async with inner(a):
            nb.start()
            reaper = asyncio.create_task(pool.reaper())
            try:
                yield
            finally:
                reaper.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await reaper
                nb.stop()
                await pool.close_all()
                db.purge_expired()

    app.router.lifespan_context = lifespan
    return PathTokenMiddleware(app)


class PathTokenMiddleware:
    """Accept `/t/<token>/mcp` for clients that cannot send an Authorization header.

    The token is moved into a Bearer header and the path rewritten to the real
    MCP endpoint, so the regular auth middleware validates it like any other.
    """
    _re = re.compile(r"^/t/([A-Za-z0-9_\-]{16,})(/mcp/?)$")

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            m = self._re.match(scope.get("path", ""))
            if m:
                token = m.group(1)
                headers = [(k, v) for k, v in scope.get("headers", []) if k.lower() != b"authorization"]
                headers.append((b"authorization", f"Bearer {token}".encode()))
                scope = dict(scope)
                scope["headers"] = headers
                scope["path"] = config.MCP_PATH
                scope["raw_path"] = config.MCP_PATH.encode()
        await self.app(scope, receive, send)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config.validate()
    import uvicorn
    app = build_app()
    print(f"{config.APP_NAME}: MCP at {config.PUBLIC_BASE_URL}{config.MCP_PATH}  "
          f"(listening on {config.MCP_HOST}:{config.MCP_PORT})", file=sys.stderr, flush=True)
    uvicorn.run(app, host=config.MCP_HOST, port=config.MCP_PORT, proxy_headers=True,
                forwarded_allow_ips="*", log_level="info")


if __name__ == "__main__":
    main()
