"""Telegram Mini App routes: page serving + initData-scoped API."""

import hashlib
import hmac
import io
import json
import time
from urllib.parse import urlencode

import telegram_notebook.main as web

BOT_TOKEN = "424242:miniapp-test-token"


def _sign(user_id: int, token: str = BOT_TOKEN) -> str:
    fields = {
        "auth_date": str(int(time.time())),
        "user": json.dumps({"id": user_id, "first_name": "Mina"}),
    }
    data_check = "\n".join(f"{k}={fields[k]}" for k in sorted(fields))
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    h = hmac.new(secret, data_check.encode(), hashlib.sha256).hexdigest()
    return urlencode({**fields, "hash": h})


class FakeGet(web.RequestHandler):
    def __init__(self, path, headers=None):
        self.path = path
        self.client_address = ("8.8.8.8", 1)
        self.headers = headers or {}
        self.html = None
        self.sent = None

    def _send_html(self, html, status=200):
        self.html = (status, html)

    def _send_json(self, payload, status=200):
        self.sent = (status, payload)


class FakePost(web.RequestHandler):
    def __init__(self, path, body=b"{}", headers=None):
        self.path = path
        self.rfile = io.BytesIO(body)
        self.headers = headers or {"Content-Length": str(len(body))}
        self.client_address = ("8.8.8.8", 1)
        self.sent = None

    def _send_json(self, payload, status=200):
        self.sent = (status, payload)


def test_miniapp_page_is_served():
    h = FakeGet("/miniapp")
    h.do_GET()
    status, html = h.html
    assert status == 200
    assert "telegram-web-app.js" in html
    assert 'id="askBtn"' in html


def test_miniapp_ask_requires_init_data():
    body = b'{"query": "hi"}'
    h = FakePost("/api/miniapp/ask", body, {"Content-Length": str(len(body))})
    h.do_POST()
    assert h.sent[0] == 401  # no initData header -> unauthorized


def test_miniapp_ask_scopes_to_verified_user(monkeypatch):
    monkeypatch.setattr(web.state.settings, "telegram_bot_token", BOT_TOKEN)

    captured = {}

    class FakeSearch:
        def search(self, *, owner_id, query, top_k):
            captured["owner_id"] = owner_id
            captured["query"] = query
            return []

        def generate_answer(self, *, query, results, **kwargs):
            return "stub answer"

    monkeypatch.setattr(web.state, "search_service", FakeSearch())

    body = json.dumps({"query": "what did I save about villas?"}).encode()
    headers = {"Content-Length": str(len(body)), "X-Telegram-Init-Data": _sign(9090)}
    h = FakePost("/api/miniapp/ask", body, headers)
    h.do_POST()

    status, payload = h.sent
    assert status == 200
    assert payload["answer"] == "stub answer"
    assert captured["owner_id"] == 9090  # scoped to the authenticated Telegram user
    assert captured["query"].startswith("what did I save")


def test_miniapp_ask_rejects_forged_init_data(monkeypatch):
    monkeypatch.setattr(web.state.settings, "telegram_bot_token", BOT_TOKEN)
    forged = _sign(123, token="wrong-token")
    body = json.dumps({"query": "x"}).encode()
    headers = {"Content-Length": str(len(body)), "X-Telegram-Init-Data": forged}
    h = FakePost("/api/miniapp/ask", body, headers)
    h.do_POST()
    assert h.sent[0] == 401
