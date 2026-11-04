"""Verify Telegram Mini App (Web App) ``initData``.

When a Mini App opens, Telegram hands the page a signed ``initData`` string. The
backend must verify that signature with the bot token before trusting the user
identity inside it. The algorithm is Telegram's documented scheme:

    secret_key   = HMAC_SHA256(key="WebAppData", message=bot_token)
    data_check   = "\n".join sorted "key=value" pairs, excluding "hash"
    expected     = hex( HMAC_SHA256(key=secret_key, message=data_check) )
    valid        = expected == provided "hash"

This is implemented with the standard library only (``hmac``/``hashlib``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from urllib.parse import parse_qsl

logger = logging.getLogger(__name__)


def verify_init_data(init_data: str, bot_token: str, *, max_age_seconds: int = 86400) -> dict | None:
    """Return the parsed Mini App fields if ``init_data`` is authentic, else None.

    ``max_age_seconds`` rejects stale payloads via ``auth_date`` (0 disables the check).
    On success the returned dict includes a parsed ``user`` dict when present.
    """
    if not init_data or not bot_token:
        return None

    pairs = dict(parse_qsl(init_data, keep_blank_values=True))
    received_hash = pairs.pop("hash", None)
    if not received_hash:
        return None

    data_check_string = "\n".join(f"{key}={pairs[key]}" for key in sorted(pairs))
    secret_key = hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()
    expected_hash = hmac.new(secret_key, data_check_string.encode("utf-8"), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(expected_hash, received_hash):
        return None

    if max_age_seconds:
        try:
            auth_date = int(pairs.get("auth_date", "0"))
        except ValueError:
            return None
        if auth_date <= 0 or (time.time() - auth_date) > max_age_seconds:
            return None

    user = None
    if "user" in pairs:
        try:
            user = json.loads(pairs["user"])
        except (ValueError, TypeError):
            user = None

    return {**pairs, "user": user}


def init_data_user_id(verified: dict | None) -> int | None:
    """Extract the Telegram user id from verified init data, or None."""
    if not verified:
        return None
    user = verified.get("user")
    if isinstance(user, dict) and "id" in user:
        try:
            return int(user["id"])
        except (ValueError, TypeError):
            return None
    return None
