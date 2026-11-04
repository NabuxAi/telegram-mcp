from telegram_notebook.telegram_client import _canonical_channel_url


def test_prefers_username():
    assert _canonical_channel_url("mychan", "https://t.me/whatever", 5) == "https://t.me/mychan"


def test_uses_raw_url_when_no_username():
    assert _canonical_channel_url(None, "https://t.me/joinchat/AbC/", 5) == "https://t.me/joinchat/AbC"


def test_falls_back_to_internal_scheme():
    assert _canonical_channel_url(None, "not-a-url", 77) == "telegram://channel/77"
