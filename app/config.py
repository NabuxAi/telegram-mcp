"""Environment-driven configuration.

Everything is read once at import time. `validate()` is called by the entrypoint
so that importing the package (e.g. from tests) never fails on missing values.
"""
import os
from pathlib import Path


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


# ---- Telegram application credentials (from my.telegram.org) ----------------
# One api_id/api_hash pair identifies THIS app; every user logs in through it
# with their own phone number, exactly like any third-party Telegram client.
TG_API_ID = _int("TG_API_ID", 0)
TG_API_HASH = os.environ.get("TG_API_HASH", "").strip()

# Let a user supply their own api_id/api_hash on the login page (advanced).
ALLOW_USER_API_CREDENTIALS = _bool("ALLOW_USER_API_CREDENTIALS", True)

# Optional allow-list of phone numbers (comma separated, with country code).
# Empty means anyone may connect.
ALLOWED_PHONES = {
    p.strip() for p in os.environ.get("ALLOWED_PHONES", "").split(",") if p.strip()
}

# ---- Public URL / secrets ----------------------------------------------------
# Public HTTPS origin of this server, no trailing slash. Used as the OAuth issuer
# and for every absolute URL shown to users.
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").strip().rstrip("/")

# Master secret: derives the key that encrypts Telegram sessions at rest and the
# key that signs browser cookies. Changing it invalidates every stored session.
APP_SECRET_KEY = os.environ.get("APP_SECRET_KEY", "").strip()

# ---- Storage -----------------------------------------------------------------
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
DB_PATH = Path(os.environ.get("DB_PATH", "") or (DATA_DIR / "app.db"))

# ---- HTTP --------------------------------------------------------------------
MCP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT = _int("MCP_PORT", 8000)
MCP_PATH = "/mcp"

# ---- Token lifetimes ---------------------------------------------------------
ACCESS_TOKEN_TTL = _int("ACCESS_TOKEN_TTL", 3600)             # 1 hour
REFRESH_TOKEN_TTL = _int("REFRESH_TOKEN_TTL", 90 * 24 * 3600)  # 90 days
AUTH_CODE_TTL = 600
LOGIN_FLOW_TTL = 900
COOKIE_TTL = _int("COOKIE_TTL", 30 * 24 * 3600)

# ---- Telegram behaviour ------------------------------------------------------
FLOODWAIT_CAP = _int("TG_FLOODWAIT_CAP", 45)
CLIENT_IDLE_SECONDS = _int("TG_CLIENT_IDLE_SECONDS", 900)
FILE_FETCH_TIMEOUT = float(os.environ.get("TG_IMAGE_TIMEOUT", "30") or 30)
MAX_UPLOAD_BYTES = _int("TG_MAX_UPLOAD_MB", 32) * 1024 * 1024
MAX_INLINE_MEDIA_BYTES = _int("TG_MAX_INLINE_MEDIA_KB", 1536) * 1024
ALLOW_PRIVATE_HOSTS = _bool("TG_ALLOW_PRIVATE_IMAGE_HOSTS", False)

# Login abuse limits (per client IP, sliding window of one hour)
LOGIN_CODES_PER_HOUR = _int("LOGIN_CODES_PER_HOUR", 10)

APP_NAME = os.environ.get("APP_NAME", "Telegram MCP").strip() or "Telegram MCP"


def validate() -> None:
    """Fail fast on misconfiguration. Called by the entrypoint only."""
    problems = []
    if not TG_API_ID or not TG_API_HASH:
        problems.append("TG_API_ID and TG_API_HASH are required (from https://my.telegram.org).")
    if not PUBLIC_BASE_URL:
        problems.append("PUBLIC_BASE_URL is required, e.g. https://tg-mcp.example.com")
    elif not (PUBLIC_BASE_URL.startswith("https://")
              or PUBLIC_BASE_URL.startswith("http://localhost")
              or PUBLIC_BASE_URL.startswith("http://127.0.0.1")):
        problems.append("PUBLIC_BASE_URL must be https:// (http is only allowed for localhost).")
    weak = {"change-me", "secret", "test", "changeme"}
    if len(APP_SECRET_KEY) < 32 or APP_SECRET_KEY.lower() in weak:
        problems.append(
            "APP_SECRET_KEY must be a random string of at least 32 characters. Generate one with:\n"
            "  python -c \"import secrets; print(secrets.token_urlsafe(48))\""
        )
    if problems:
        raise SystemExit("Configuration errors:\n- " + "\n- ".join(problems))
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise SystemExit(f"Cannot create DATA_DIR {DATA_DIR}: {e}. Mount a writable volume there.") from e


def user_dir(user_id: int) -> Path:
    p = DATA_DIR / "users" / str(user_id)
    p.mkdir(parents=True, exist_ok=True)
    return p
