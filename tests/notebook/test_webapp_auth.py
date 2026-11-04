"""Telegram Mini App initData signature verification."""

import hashlib
import hmac
import json
import time
from urllib.parse import urlencode

from telegram_notebook.webapp_auth import init_data_user_id, verify_init_data

BOT_TOKEN = "123456:TEST-bot-token"


def _sign(fields: dict, token: str = BOT_TOKEN) -> str:
    """Build a correctly signed initData query string for tests."""
    data_check_string = "\n".join(f"{k}={fields[k]}" for k in sorted(fields))
    secret_key = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    h = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
    return urlencode({**fields, "hash": h})


def test_valid_init_data_round_trips():
    fields = {
        "auth_date": str(int(time.time())),
        "query_id": "abc",
        "user": json.dumps({"id": 777, "first_name": "Sam"}),
    }
    verified = verify_init_data(_sign(fields), BOT_TOKEN)
    assert verified is not None
    assert verified["user"]["first_name"] == "Sam"
    assert init_data_user_id(verified) == 777


def test_tampered_data_is_rejected():
    fields = {"auth_date": str(int(time.time())), "user": json.dumps({"id": 1})}
    signed = _sign(fields)
    tampered = signed.replace("%22id%22%3A+1", "%22id%22%3A+2") if "id" in signed else signed + "&x=1"
    assert verify_init_data(tampered, BOT_TOKEN) is None


def test_wrong_token_is_rejected():
    fields = {"auth_date": str(int(time.time())), "user": json.dumps({"id": 1})}
    assert verify_init_data(_sign(fields, token="other:token"), BOT_TOKEN) is None


def test_missing_hash_is_rejected():
    assert verify_init_data("auth_date=1&user=%7B%7D", BOT_TOKEN) is None


def test_stale_auth_date_is_rejected():
    fields = {"auth_date": str(int(time.time()) - 100000), "user": json.dumps({"id": 1})}
    assert verify_init_data(_sign(fields), BOT_TOKEN, max_age_seconds=3600) is None
    # but allowed when the freshness check is disabled
    assert verify_init_data(_sign(fields), BOT_TOKEN, max_age_seconds=0) is not None


def test_empty_inputs():
    assert verify_init_data("", BOT_TOKEN) is None
    assert verify_init_data("auth_date=1&hash=x", "") is None
    assert init_data_user_id(None) is None
