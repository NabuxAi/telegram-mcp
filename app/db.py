"""SQLite persistence: users (encrypted Telegram sessions), OAuth clients,
authorization codes and tokens.

sqlite3 from the standard library is enough here: every query is a point lookup
and the server is I/O bound on Telegram, not on the database. A single lock
serializes writes; reads share the same connection (check_same_thread=False).
"""
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional

from . import config, security

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    tg_user_id    INTEGER UNIQUE NOT NULL,
    phone         TEXT,
    first_name    TEXT,
    last_name     TEXT,
    username      TEXT,
    session_enc   TEXT NOT NULL,
    api_id        INTEGER,
    api_hash_enc  TEXT,
    session_ok    INTEGER NOT NULL DEFAULT 1,
    created_at    INTEGER NOT NULL,
    last_seen_at  INTEGER
);
CREATE TABLE IF NOT EXISTS oauth_clients (
    client_id     TEXT PRIMARY KEY,
    data_json     TEXT NOT NULL,
    created_at    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS auth_codes (
    code          TEXT PRIMARY KEY,
    client_id     TEXT NOT NULL,
    user_id       INTEGER NOT NULL,
    data_json     TEXT NOT NULL,
    expires_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tokens (
    token_hash    TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,            -- 'access' | 'refresh'
    client_id     TEXT NOT NULL,
    user_id       INTEGER NOT NULL,
    scopes        TEXT NOT NULL,
    pair_id       TEXT NOT NULL,            -- links an access token to its refresh token
    label         TEXT,
    expires_at    INTEGER,                  -- NULL = never
    revoked       INTEGER NOT NULL DEFAULT 0,
    created_at    INTEGER NOT NULL,
    last_used_at  INTEGER
);
CREATE INDEX IF NOT EXISTS tokens_user ON tokens(user_id);
CREATE INDEX IF NOT EXISTS tokens_pair ON tokens(pair_id);
"""


class Database:
    def __init__(self, path: Optional[str] = None):
        self.path = str(path or config.DB_PATH)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(_SCHEMA)   # executescript manages its own transaction
            self._migrate_accounts()

    def _migrate_accounts(self) -> None:
        """One person, several Telegram accounts.

        `users` was one row per Telegram account with nothing above it, so a
        person who connected two phones was two unrelated identities. owner_key
        groups the rows that belong to one person; position is the order they
        connected them, which is the order the API lists them in.

        Idempotent. Existing rows become their own owner ("tg:<id>"), so nothing
        changes for anyone until they connect a second account.
        """
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(users)").fetchall()}
        if "owner_key" not in cols:
            self._conn.execute("ALTER TABLE users ADD COLUMN owner_key TEXT")
        if "position" not in cols:
            self._conn.execute("ALTER TABLE users ADD COLUMN position INTEGER NOT NULL DEFAULT 0")
        self._conn.execute(
            "UPDATE users SET owner_key = 'tg:' || tg_user_id WHERE owner_key IS NULL OR owner_key = ''"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS users_owner ON users(owner_key, position)")

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def _q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def _one(self, sql: str, args: tuple = ()) -> Optional[sqlite3.Row]:
        rows = self._q(sql, args)
        return rows[0] if rows else None

    def close(self):
        with self._lock:
            self._conn.close()

    # ---- users ----------------------------------------------------------
    def upsert_user(self, *, tg_user_id: int, phone: str, first_name: str, last_name: str,
                    username: Optional[str], session_string: str,
                    api_id: Optional[int] = None, api_hash: Optional[str] = None,
                    owner_key: Optional[str] = None) -> int:
        """Create or refresh the row for one Telegram account.

        owner_key is who this account belongs to. Given when someone who is
        already signed in connects another phone — the new account joins their
        existing set. Absent, the account is its own owner, which is exactly the
        old behaviour.

        An account that already exists is never moved to a different owner here:
        re-logging into a phone you already connected must not hand it to
        whoever happens to hold the browser session.
        """
        now = int(time.time())
        enc = security.encrypt(session_string)
        hash_enc = security.encrypt(api_hash) if api_hash else None
        with self._tx() as c:
            row = c.execute("SELECT id FROM users WHERE tg_user_id=?", (tg_user_id,)).fetchone()
            if row:
                c.execute(
                    "UPDATE users SET phone=?, first_name=?, last_name=?, username=?, session_enc=?, "
                    "api_id=?, api_hash_enc=?, session_ok=1, last_seen_at=? WHERE id=?",
                    (phone, first_name, last_name, username, enc, api_id, hash_enc, now, row["id"]),
                )
                return int(row["id"])
            owner = owner_key or f"tg:{tg_user_id}"
            nxt = c.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 FROM users WHERE owner_key=?", (owner,)
            ).fetchone()[0]
            cur = c.execute(
                "INSERT INTO users (tg_user_id, phone, first_name, last_name, username, session_enc, "
                "api_id, api_hash_enc, session_ok, created_at, last_seen_at, owner_key, position) "
                "VALUES (?,?,?,?,?,?,?,?,1,?,?,?,?)",
                (tg_user_id, phone, first_name, last_name, username, enc, api_id, hash_enc, now, now,
                 owner, int(nxt)),
            )
            return int(cur.lastrowid)

    def accounts_for_owner(self, owner_key: str) -> list[dict]:
        """Every account this person connected, in the order they connected them.

        Order is the contract: "the first connected account" is what a tool acts
        on when the caller names none, so it must be stable across calls.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM users WHERE owner_key=? ORDER BY position, id", (owner_key,)
            ).fetchall()
        return [dict(r) for r in rows]

    def owner_of(self, user_id: int) -> Optional[str]:
        u = self.get_user(user_id)
        return (u or {}).get("owner_key") or None

    def get_user(self, user_id: int) -> Optional[dict]:
        row = self._one("SELECT * FROM users WHERE id=?", (user_id,))
        return dict(row) if row else None

    def get_user_by_tg_id(self, tg_user_id: int) -> Optional[dict]:
        row = self._one("SELECT * FROM users WHERE tg_user_id=?", (tg_user_id,))
        return dict(row) if row else None

    def user_session(self, user_id: int) -> tuple[str, int, str]:
        """Return (session_string, api_id, api_hash) for a user."""
        u = self.get_user(user_id)
        if not u:
            raise KeyError(f"user {user_id} not found")
        session = security.decrypt(u["session_enc"])
        api_id = int(u["api_id"] or config.TG_API_ID)
        api_hash = security.decrypt(u["api_hash_enc"]) if u["api_hash_enc"] else config.TG_API_HASH
        return session, api_id, api_hash

    def update_session(self, user_id: int, session_string: str) -> None:
        with self._tx() as c:
            c.execute("UPDATE users SET session_enc=?, session_ok=1 WHERE id=?",
                      (security.encrypt(session_string), user_id))

    def mark_session(self, user_id: int, ok: bool) -> None:
        with self._tx() as c:
            c.execute("UPDATE users SET session_ok=? WHERE id=?", (1 if ok else 0, user_id))

    def touch_user(self, user_id: int) -> None:
        with self._tx() as c:
            c.execute("UPDATE users SET last_seen_at=? WHERE id=?", (int(time.time()), user_id))

    def delete_user(self, user_id: int) -> None:
        with self._tx() as c:
            c.execute("DELETE FROM tokens WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM auth_codes WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM users WHERE id=?", (user_id,))

    # ---- oauth clients --------------------------------------------------
    def save_client(self, client_id: str, data: dict) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO oauth_clients (client_id, data_json, created_at) VALUES (?,?,?)",
                (client_id, json.dumps(data), int(time.time())),
            )

    def get_client(self, client_id: str) -> Optional[dict]:
        row = self._one("SELECT data_json FROM oauth_clients WHERE client_id=?", (client_id,))
        return json.loads(row["data_json"]) if row else None

    # ---- auth codes -----------------------------------------------------
    def save_auth_code(self, code: str, client_id: str, user_id: int, data: dict, expires_at: float) -> None:
        with self._tx() as c:
            c.execute("DELETE FROM auth_codes WHERE expires_at < ?", (time.time(),))
            c.execute(
                "INSERT INTO auth_codes (code, client_id, user_id, data_json, expires_at) VALUES (?,?,?,?,?)",
                (code, client_id, user_id, json.dumps(data), expires_at),
            )

    def pop_auth_code(self, code: str) -> Optional[dict]:
        """Return and delete the code (single use)."""
        with self._tx() as c:
            row = c.execute("SELECT * FROM auth_codes WHERE code=?", (code,)).fetchone()
            if not row:
                return None
            c.execute("DELETE FROM auth_codes WHERE code=?", (code,))
        d = json.loads(row["data_json"])
        d.update({"code": row["code"], "client_id": row["client_id"],
                  "user_id": row["user_id"], "expires_at": row["expires_at"]})
        return d

    def peek_auth_code(self, code: str) -> Optional[dict]:
        row = self._one("SELECT * FROM auth_codes WHERE code=?", (code,))
        if not row:
            return None
        d = json.loads(row["data_json"])
        d.update({"code": row["code"], "client_id": row["client_id"],
                  "user_id": row["user_id"], "expires_at": row["expires_at"]})
        return d

    # ---- tokens ---------------------------------------------------------
    def save_token(self, token: str, *, kind: str, client_id: str, user_id: int, scopes: list[str],
                   pair_id: str, expires_at: Optional[int], label: Optional[str] = None) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT INTO tokens (token_hash, kind, client_id, user_id, scopes, pair_id, label, "
                "expires_at, revoked, created_at) VALUES (?,?,?,?,?,?,?,?,0,?)",
                (security.hash_token(token), kind, client_id, user_id, " ".join(scopes),
                 pair_id, label, expires_at, int(time.time())),
            )

    def get_token(self, token: str, kind: Optional[str] = None) -> Optional[dict]:
        row = self._one("SELECT * FROM tokens WHERE token_hash=?", (security.hash_token(token),))
        if not row:
            return None
        d = dict(row)
        if kind and d["kind"] != kind:
            return None
        if d["revoked"]:
            return None
        if d["expires_at"] and d["expires_at"] < time.time():
            return None
        d["scopes"] = d["scopes"].split() if d["scopes"] else []
        return d

    def touch_token(self, token: str) -> None:
        with self._tx() as c:
            c.execute("UPDATE tokens SET last_used_at=? WHERE token_hash=?",
                      (int(time.time()), security.hash_token(token)))

    def revoke_pair(self, pair_id: str) -> None:
        with self._tx() as c:
            c.execute("UPDATE tokens SET revoked=1 WHERE pair_id=?", (pair_id,))

    def revoke_by_hash(self, token_hash: str, user_id: int) -> bool:
        with self._tx() as c:
            row = c.execute("SELECT pair_id FROM tokens WHERE token_hash=? AND user_id=?",
                            (token_hash, user_id)).fetchone()
            if not row:
                return False
            c.execute("UPDATE tokens SET revoked=1 WHERE pair_id=?", (row["pair_id"],))
            return True

    def list_tokens(self, user_id: int) -> list[dict]:
        rows = self._q(
            "SELECT t.*, c.data_json AS client_json FROM tokens t "
            "LEFT JOIN oauth_clients c ON c.client_id = t.client_id "
            "WHERE t.user_id=? AND t.revoked=0 AND (t.expires_at IS NULL OR t.expires_at > ?) "
            "ORDER BY t.created_at DESC",
            (user_id, int(time.time())),
        )
        out = []
        for r in rows:
            d = dict(r)
            client = json.loads(d.pop("client_json") or "{}")
            d["client_name"] = client.get("client_name") or d["client_id"]
            out.append(d)
        return out

    def purge_expired(self) -> None:
        with self._tx() as c:
            c.execute("DELETE FROM tokens WHERE revoked=1 OR (expires_at IS NOT NULL AND expires_at < ?)",
                      (int(time.time()) - 7 * 24 * 3600,))
            c.execute("DELETE FROM auth_codes WHERE expires_at < ?", (time.time(),))


_db: Optional[Database] = None


def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db


def set_db(db: Database) -> None:
    global _db
    _db = db
