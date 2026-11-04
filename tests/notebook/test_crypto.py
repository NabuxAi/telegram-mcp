from telegram_notebook import crypto

KEY = "test-master-key-please-change"


def test_roundtrip_with_key(monkeypatch):
    monkeypatch.setenv("SECRETS_KEY", KEY)
    token = crypto.encrypt("super-secret-session")
    assert token != "super-secret-session"
    assert token.startswith("enc::")
    assert crypto.decrypt(token) == "super-secret-session"


def test_encrypt_is_nondeterministic(monkeypatch):
    monkeypatch.setenv("SECRETS_KEY", KEY)
    a = crypto.encrypt("same value")
    b = crypto.encrypt("same value")
    assert a != b  # random nonce per message
    assert crypto.decrypt(a) == crypto.decrypt(b) == "same value"


def test_none_and_empty_passthrough(monkeypatch):
    monkeypatch.setenv("SECRETS_KEY", KEY)
    assert crypto.encrypt(None) is None
    assert crypto.encrypt("") == ""
    assert crypto.decrypt(None) is None
    assert crypto.decrypt("") == ""


def test_legacy_plaintext_passthrough(monkeypatch):
    monkeypatch.setenv("SECRETS_KEY", KEY)
    # Values without the enc:: prefix are treated as legacy plaintext.
    assert crypto.decrypt("plain-legacy-value") == "plain-legacy-value"


def test_double_encrypt_is_noop(monkeypatch):
    monkeypatch.setenv("SECRETS_KEY", KEY)
    once = crypto.encrypt("value")
    assert crypto.encrypt(once) == once


def test_tampering_is_rejected(monkeypatch):
    monkeypatch.setenv("SECRETS_KEY", KEY)
    token = crypto.encrypt("value")
    tampered = token[:-2] + ("AA" if not token.endswith("AA") else "BB")
    assert crypto.decrypt(tampered) is None


def test_wrong_key_cannot_decrypt(monkeypatch):
    monkeypatch.setenv("SECRETS_KEY", KEY)
    token = crypto.encrypt("value")
    monkeypatch.setenv("SECRETS_KEY", "a-different-key")
    assert crypto.decrypt(token) is None


def test_no_key_is_noop_and_cannot_read_encrypted(monkeypatch):
    monkeypatch.setenv("SECRETS_KEY", KEY)
    token = crypto.encrypt("value")
    monkeypatch.delenv("SECRETS_KEY", raising=False)
    # Without a key, encryption is a transparent no-op...
    assert crypto.encrypt("value") == "value"
    assert not crypto.encryption_enabled()
    # ...and an existing encrypted value cannot be read.
    assert crypto.decrypt(token) is None
