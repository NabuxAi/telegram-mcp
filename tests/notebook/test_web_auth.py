import telegram_notebook.main as web


class FakeHandler(web.RequestHandler):
    """Exercises the auth helpers without standing up a real HTTP server."""

    def __init__(self, headers=None, client="127.0.0.1"):
        self.headers = headers or {}
        self.client_address = (client, 12345)
        self.sent = None

    def _send_json(self, payload, status=200):
        self.sent = (status, payload)


def test_token_required_and_accepts_bearer(monkeypatch):
    monkeypatch.setattr(web.state.settings, "web_api_token", "s3cret")
    h = FakeHandler(headers={"Authorization": "Bearer s3cret"}, client="8.8.8.8")
    assert h._require_auth() is True
    assert h.sent is None


def test_token_accepts_x_api_token_header(monkeypatch):
    monkeypatch.setattr(web.state.settings, "web_api_token", "s3cret")
    h = FakeHandler(headers={"X-API-Token": "s3cret"}, client="8.8.8.8")
    assert h._require_auth() is True


def test_wrong_token_is_rejected(monkeypatch):
    monkeypatch.setattr(web.state.settings, "web_api_token", "s3cret")
    h = FakeHandler(headers={"X-API-Token": "nope"}, client="8.8.8.8")
    assert h._require_auth() is False
    assert h.sent[0] == 401


def test_missing_token_is_rejected(monkeypatch):
    monkeypatch.setattr(web.state.settings, "web_api_token", "s3cret")
    h = FakeHandler(headers={}, client="8.8.8.8")
    assert h._require_auth() is False
    assert h.sent[0] == 401


def test_no_token_allows_loopback_only(monkeypatch):
    monkeypatch.setattr(web.state.settings, "web_api_token", None)
    assert FakeHandler(client="127.0.0.1")._require_auth() is True
    assert FakeHandler(client="::1")._require_auth() is True

    remote = FakeHandler(client="10.0.0.5")
    assert remote._require_auth() is False
    assert remote.sent[0] == 401
