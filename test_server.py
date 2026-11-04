#!/usr/bin/env python3
"""Offline tests for the multi-user Telegram MCP server.

No network and no Telegram account: Telethon is replaced with a fake client and
everything runs in-process through an ASGI test client. Covers config, crypto,
the token store, the full OAuth 2.1 + Telegram-login round trip, MCP tool calls
scoped to the logged-in user, the path-token fallback and serialisation helpers.

Run:  python -m pytest -q test_server.py
"""
import asyncio
import base64
import hashlib
import json
import os
import secrets
import tempfile
from datetime import datetime, timezone
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

_TMP = tempfile.mkdtemp(prefix="tgmcp_test_")
os.environ.update({
    "TG_API_ID": "1", "TG_API_HASH": "h" * 32,
    "PUBLIC_BASE_URL": "http://localhost:8000",
    "APP_SECRET_KEY": secrets.token_urlsafe(48),
    "DATA_DIR": _TMP, "DB_PATH": os.path.join(_TMP, "store.db"),
    "LLM_PROVIDER": "openai",           # no key -> notebook_ask reports "no LLM"
    "EMBEDDING_PROVIDER": "openai",     # no key -> keyword search path
})

import httpx  # noqa: E402

import server  # noqa: E402
from app import config, security, tg  # noqa: E402
from app.db import Database, set_db  # noqa: E402


# ----------------------------------------------------------------------------
# Fake Telethon
# ----------------------------------------------------------------------------
def FakeUser(id, first_name="Ali", last_name="", username="ali", phone="989120000000", bot=False):
    from telethon.tl.types import User
    return User(id=id, first_name=first_name, last_name=last_name or None, username=username, phone=phone, bot=bot)


class FakeMessage:
    def __init__(self, mid, text, peer_id, sender):
        self.id, self.message, self.peer_id, self.sender = mid, text, peer_id, sender
        self.sender_id = sender.id
        self.date = datetime(2026, 9, 1, tzinfo=timezone.utc)
        self.out = False
        self.media = self.photo = self.document = self.web_preview = self.poll = self.contact = self.geo = self.dice = None
        self.reply_to_msg_id = self.views = self.forwards = self.edit_date = self.grouped_id = None
        self.pinned = self.mentioned = False
        self.reactions = self.fwd_from = None
        self.via_bot_id = self.action = None


class FakeSession:
    def __init__(self, s):
        self._s = s or "SESSION-" + secrets.token_hex(4)

    def save(self):
        return self._s


class FakeTG:
    """Just enough surface for login + a couple of tools."""
    instances: list = []
    password_required = False

    def __init__(self, session, api_id, api_hash):
        sess = session if isinstance(session, str) else (session.save() or "")
        self.session = FakeSession(sess)
        self.api_id, self.api_hash = api_id, api_hash
        self._connected = False
        self._authorized = sess.startswith("SESSION-")
        self.me = FakeUser(4242)
        self.logged_out = False
        FakeTG.instances.append(self)

    async def connect(self):
        self._connected = True

    async def disconnect(self):
        self._connected = False

    def is_connected(self):
        return self._connected

    async def is_user_authorized(self):
        return self._authorized

    async def send_code_request(self, phone):
        return SimpleNamespace(phone_code_hash="pch")

    async def sign_in(self, phone=None, code=None, password=None, phone_code_hash=None):
        from telethon.errors import PhoneCodeInvalidError, SessionPasswordNeededError
        if password is not None:
            if password != "pw":
                from telethon.errors import PasswordHashInvalidError
                raise PasswordHashInvalidError(request=None)
            self._authorized = True
            return self.me
        if code != "12345":
            raise PhoneCodeInvalidError(request=None)
        if FakeTG.password_required:
            raise SessionPasswordNeededError(request=None)
        self._authorized = True
        return self.me

    async def get_me(self):
        return self.me

    async def log_out(self):
        self.logged_out = True
        self._authorized = False

    async def get_entity(self, ref):
        if ref == "me":
            return self.me
        if ref in ("@bob", 7):
            return FakeUser(7, "Bob", username="bob")
        raise ValueError(f"no entity {ref!r}")

    async def get_messages(self, ent, ids=None, **kw):
        from telethon.tl.types import PeerUser
        peer = PeerUser(user_id=ent.id)
        if isinstance(ids, list):
            return [FakeMessage(i, f"msg {i}", peer, ent) for i in ids]
        return FakeMessage(ids, f"msg {ids}", peer, ent)


@pytest.fixture(scope="module")
def db():
    d = Database(os.path.join(_TMP, "store.db"))
    set_db(d)
    return d


@pytest.fixture(scope="module")
def app(db, request):
    tg._new_client = FakeTG  # type: ignore[assignment]
    application = server.build_app()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # anyio task groups must be entered and exited from the same task, so the
    # lifespan is held open by one background task for the whole module.
    stop = asyncio.Event()
    ready = asyncio.Event()

    async def hold():
        async with application.app.router.lifespan_context(application.app):
            ready.set()
            await stop.wait()

    holder = loop.create_task(hold())
    loop.run_until_complete(ready.wait())

    def _teardown():
        stop.set()
        loop.run_until_complete(holder)
        loop.close()

    request.addfinalizer(_teardown)
    return application, loop


def run(loop, coro):
    return loop.run_until_complete(coro)


def client_for(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8000")


# ----------------------------------------------------------------------------
# Unit-level
# ----------------------------------------------------------------------------
def test_crypto_roundtrip_and_cookie():
    enc = security.encrypt("s3cret")
    assert enc != "s3cret" and security.decrypt(enc) == "s3cret"
    c = security.sign_cookie(5)
    assert security.verify_cookie(c) == 5
    assert security.verify_cookie(c[:-2] + "zz") is None
    assert security.verify_cookie("junk") is None
    assert security.check_csrf(c, security.csrf_token(c)) and not security.check_csrf(c, "nope")


def test_normalize_target_and_phone():
    assert tg.normalize_target("-1001234567890") == -1001234567890
    assert tg.normalize_target("Me") == "me"
    assert tg.normalize_target("https://t.me/foo") == "@foo"
    assert tg.normalize_target("+989121234567") == "+989121234567"
    assert tg.normalize_phone("00 98 912 123 4567") == "+989121234567"
    with pytest.raises(tg.LoginError):
        tg.normalize_phone("12")


def test_token_store(db):
    uid = db.upsert_user(tg_user_id=1, phone="+1", first_name="A", last_name="", username=None, session_string="S")
    assert db.user_session(uid)[0] == "S"
    db.save_token("tok", kind="access", client_id="c", user_id=uid, scopes=["telegram"], pair_id="p", expires_at=None)
    assert db.get_token("tok")["user_id"] == uid
    assert db.get_token("tok", kind="refresh") is None
    db.revoke_pair("p")
    assert db.get_token("tok") is None
    db.save_token("old", kind="access", client_id="c", user_id=uid, scopes=[], pair_id="q", expires_at=1)
    assert db.get_token("old") is None


def test_serializers():
    u = FakeUser(9, "Zed", username="zed")
    d = tg.ser_entity(u)
    assert d["kind"] == "user" and d["title"] == "Zed" and d["chat_id"] == 9
    from telethon.tl.types import PeerUser
    m = FakeMessage(3, "hello world", PeerUser(user_id=9), u)
    s = tg.ser_message(m, text_limit=5)
    assert s["text"] == "hello…" and s["text_len"] == 11 and s["sender"]["name"] == "Zed"
    assert tg.parse_mode_of("HTML") == "html" and tg.parse_mode_of("none") is None
    assert tg.parse_datetime("2026-01-01T10:00:00").tzinfo is not None


def test_source_ref():
    from app.notebook import source_ref
    assert source_ref("@foo") == "https://t.me/foo"
    assert source_ref("-100123") == "telegram://channel/-100123"
    assert source_ref("inbox") == "inbox://forwarded"
    assert source_ref("https://t.me/foo/") == "https://t.me/foo"


# ----------------------------------------------------------------------------
# Full OAuth + Telegram login + MCP call round trip
# ----------------------------------------------------------------------------
def _pkce():
    verifier = secrets.token_urlsafe(40)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    return verifier, challenge


def _mcp_call(client, token, method, params=None, id=1):
    return client.post("/mcp", json={"jsonrpc": "2.0", "id": id, "method": method, "params": params or {}},
                       headers={"Authorization": f"Bearer {token}", "Accept": "application/json, text/event-stream"})


def test_oauth_login_and_tool_call(app):
    application, loop = app

    async def flow():
        async with client_for(application) as c:
            meta = (await c.get("/.well-known/oauth-authorization-server")).json()
            assert meta["authorization_endpoint"].endswith("/authorize")
            assert (await c.get("/healthz")).text == "ok"
            assert (await c.get("/")).status_code == 200

            # unauthenticated MCP -> 401 with resource metadata hint
            r = await c.post("/mcp", json={"jsonrpc": "2.0", "id": 0, "method": "tools/list"},
                             headers={"Accept": "application/json, text/event-stream"})
            assert r.status_code == 401 and "resource_metadata" in r.headers.get("www-authenticate", "")

            # dynamic client registration
            reg = (await c.post("/register", json={"client_name": "Claude", "redirect_uris": ["https://claude.ai/cb"],
                                                   "token_endpoint_auth_method": "none"})).json()
            cid = reg["client_id"]
            verifier, challenge = _pkce()
            r = await c.get("/authorize", params={"response_type": "code", "client_id": cid, "redirect_uri": "https://claude.ai/cb",
                                                  "code_challenge": challenge, "code_challenge_method": "S256",
                                                  "state": "xyz", "scope": "telegram"})
            assert r.status_code in (302, 307), r.text
            login_url = r.headers["location"]
            assert "/login?txn=" in login_url
            txn = parse_qs(urlsplit(login_url).query)["txn"][0]
            assert (await c.get(login_url)).status_code == 200

            # phone -> code (wrong, then right)
            r = await c.post("/login/phone", data={"phone": "+989120000000", "txn": txn})
            assert "name=\"flow\"" in r.text
            flow_id = r.text.split('name="flow" value="')[1].split('"')[0]
            r = await c.post("/login/code", data={"flow": flow_id, "code": "00000"})
            assert "Wrong code" in r.text
            r = await c.post("/login/code", data={"flow": flow_id, "code": "12345"})
            assert r.status_code == 302, r.text
            cb = r.headers["location"]
            assert cb.startswith("https://claude.ai/cb?") and "state=xyz" in cb
            code = parse_qs(urlsplit(cb).query)["code"][0]
            assert "tgmcp_session" in r.headers.get("set-cookie", "")

            # code -> tokens (PKCE)
            tok = (await c.post("/token", data={"grant_type": "authorization_code", "code": code, "client_id": cid,
                                                "redirect_uri": "https://claude.ai/cb", "code_verifier": verifier})).json()
            assert "access_token" in tok and tok["refresh_token"], tok
            # replaying the code fails
            r2 = await c.post("/token", data={"grant_type": "authorization_code", "code": code, "client_id": cid,
                                              "redirect_uri": "https://claude.ai/cb", "code_verifier": verifier})
            assert r2.status_code == 400

            # MCP: list tools, call telegram_me, call notebook_stats
            r = await _mcp_call(c, tok["access_token"], "tools/list")
            names = {t["name"] for t in r.json()["result"]["tools"]}
            assert {"telegram_me", "telegram_send_message", "notebook_ask", "notebook_import_chat"} <= names
            r = await _mcp_call(c, tok["access_token"], "tools/call", {"name": "telegram_me", "arguments": {}})
            body = json.loads(r.json()["result"]["content"][0]["text"])
            assert body["ok"] and body["account"]["chat_id"] == 4242, body
            r = await _mcp_call(c, tok["access_token"], "tools/call", {"name": "telegram_get_messages",
                                                                       "arguments": {"chat": "@bob", "ids": [1, 2]}})
            body = json.loads(r.json()["result"]["content"][0]["text"])
            assert body["ok"] and body["count"] == 2 and body["messages"][0]["text"] == "msg 1"
            r = await _mcp_call(c, tok["access_token"], "tools/call", {"name": "notebook_stats", "arguments": {}})
            body = json.loads(r.json()["result"]["content"][0]["text"])
            assert body["ok"] and body["stats"]["items"] == 0
            r = await _mcp_call(c, tok["access_token"], "tools/call", {"name": "notebook_save_note",
                                                                       "arguments": {"text": "Al Mouj has townhouses", "title": "t"}})
            body = json.loads(r.json()["result"]["content"][0]["text"])
            assert body["ok"] and body["saved"], body
            r = await _mcp_call(c, tok["access_token"], "tools/call", {"name": "notebook_search",
                                                                       "arguments": {"query": "townhouses"}})
            body = json.loads(r.json()["result"]["content"][0]["text"])
            assert body["ok"] and body["count"] == 1 and "townhouses" in body["results"][0]["chunk_text"]
            r = await _mcp_call(c, tok["access_token"], "tools/call", {"name": "notebook_ask",
                                                                       "arguments": {"question": "townhouses?"}})
            body = json.loads(r.json()["result"]["content"][0]["text"])
            assert not body["ok"] and "No LLM" in body["error"]

            # refresh rotates tokens; old access token dies
            new = (await c.post("/token", data={"grant_type": "refresh_token", "refresh_token": tok["refresh_token"],
                                                "client_id": cid})).json()
            assert new["access_token"] != tok["access_token"]
            r = await _mcp_call(c, tok["access_token"], "tools/list")
            assert r.status_code == 401
            r = await _mcp_call(c, new["access_token"], "tools/list")
            assert r.status_code == 200

            # dashboard: personal token + path-token fallback
            cookie = c.cookies.get("tgmcp_session")
            csrf = security.csrf_token(cookie)
            r = await c.post("/dashboard/token", data={"csrf": csrf, "label": "cli"})
            assert r.status_code == 200 and "/t/" in r.text
            personal = r.text.split("/t/")[1].split("/mcp")[0]
            r = await c.post(f"/t/{personal}/mcp", json={"jsonrpc": "2.0", "id": 9, "method": "tools/list"},
                             headers={"Accept": "application/json, text/event-stream"})
            assert r.status_code == 200 and r.json()["result"]["tools"]
            # revoke it
            h = security.hash_token(personal)
            r = await c.post("/dashboard/revoke", data={"csrf": csrf, "h": h})
            assert r.status_code == 303
            r = await c.post(f"/t/{personal}/mcp", json={"jsonrpc": "2.0", "id": 9, "method": "tools/list"},
                             headers={"Accept": "application/json, text/event-stream"})
            assert r.status_code == 401
            # bad csrf rejected
            r = await c.post("/dashboard/token", data={"csrf": "bad"})
            assert r.status_code == 302

            # notebook mirror: bot_users row created for this account
            from telegram_notebook.db import Repository, connect
            repo = Repository(connect(config.DB_PATH))
            bu = repo.get_bot_user(bot_user_id=4242)
            assert bu and bu["session_string"].startswith("SESSION-"), bu

            # disconnect: logs out of Telegram, deletes user, tokens die
            r = await c.post("/dashboard/disconnect", data={"csrf": csrf})
            assert r.status_code == 200 and any(i.logged_out for i in FakeTG.instances)
            r = await _mcp_call(c, new["access_token"], "tools/list")
            assert r.status_code == 401
            assert repo.get_bot_user(bot_user_id=4242)["session_string"] in (None, "")

    run(loop, flow())


def test_two_step_password_and_consent(app):
    application, loop = app
    FakeTG.password_required = True

    async def flow():
        try:
            async with client_for(application) as c:
                r = await c.post("/login/phone", data={"phone": "+989120000001"})
                flow_id = r.text.split('name="flow" value="')[1].split('"')[0]
                r = await c.post("/login/code", data={"flow": flow_id, "code": "12345"})
                assert "/login/password" in r.text
                r = await c.post("/login/password", data={"flow": flow_id, "password": "wrong"})
                assert "Wrong two-step" in r.text
                r = await c.post("/login/password", data={"flow": flow_id, "password": "pw"})
                assert r.status_code == 302 and r.headers["location"] == "/dashboard"
                assert (await c.get("/dashboard")).status_code == 200

                # second app connects: consent screen for the already-logged-in user
                reg = (await c.post("/register", json={"client_name": "Cursor", "redirect_uris": ["http://localhost:1/cb"],
                                                       "token_endpoint_auth_method": "none"})).json()
                verifier, challenge = _pkce()
                r = await c.get("/authorize", params={"response_type": "code", "client_id": reg["client_id"],
                                                      "redirect_uri": "http://localhost:1/cb", "code_challenge": challenge,
                                                      "code_challenge_method": "S256", "state": "s2"})
                txn = parse_qs(urlsplit(r.headers["location"]).query)["txn"][0]
                r = await c.get(f"/login?txn={txn}")
                assert "Cursor" in r.text and "/login/consent" in r.text
                csrf = security.csrf_token(c.cookies.get("tgmcp_session"))
                r = await c.post("/login/consent", data={"txn": txn, "csrf": csrf, "decision": "deny"})
                assert "error=access_denied" in r.headers["location"]
        finally:
            FakeTG.password_required = False

    run(loop, flow())


def test_rate_limit_and_allowlist(app):
    application, loop = app

    async def flow():
        async with client_for(application) as c:
            config.ALLOWED_PHONES.add("+10000000000")
            try:
                r = await c.post("/login/phone", data={"phone": "+989120000009"})
                assert "not allowed" in r.text
            finally:
                config.ALLOWED_PHONES.clear()
            old = config.LOGIN_CODES_PER_HOUR
            config.LOGIN_CODES_PER_HOUR = 1
            try:
                await c.post("/login/phone", data={"phone": "+989120000002"}, headers={"x-forwarded-for": "9.9.9.9"})
                r = await c.post("/login/phone", data={"phone": "+989120000002"}, headers={"x-forwarded-for": "9.9.9.9"})
                assert "زیادی" in r.text
            finally:
                config.LOGIN_CODES_PER_HOUR = old

    run(loop, flow())


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", __file__]))
