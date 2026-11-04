from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .crypto import decrypt as _decrypt
from .crypto import encrypt as _encrypt
from .keyword_search import query_terms

# Columns holding secrets that are encrypted at rest (see crypto.py).
_BOT_USER_SECRETS = ("api_hash", "session_string", "gemini_api_key")
_AUTH_FLOW_SECRETS = ("api_hash", "session_string", "phone_code_hash")


def _decrypt_row(row: dict[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any] | None:
    if row is None:
        return None
    for key in keys:
        if key in row:
            row[key] = _decrypt(row[key])
    return row


def connect(db_path: Path) -> Path:
    path = Path(db_path)
    if path.suffix.lower() != ".json":
        return path

    target = path.with_suffix(".db")
    if target.exists():
        return target

    _migrate_json_store(path, target)
    return target


def _migrate_json_store(source: Path, target: Path) -> None:
    repo = Repository(target)
    repo.init()

    if not source.exists():
        return

    raw = source.read_text(encoding="utf-8").strip()
    if not raw:
        return

    data = json.loads(raw)
    if not isinstance(data, dict):
        return

    with sqlite3.connect(target) as conn:
        for table in ("channels", "messages", "media_items", "chunks", "bot_users", "auth_flows"):
            rows = data.get(table, [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict) or not row:
                    continue
                columns = list(row.keys())
                placeholders = ", ".join("?" for _ in columns)
                col_sql = ", ".join(columns)
                values = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in row.values()]
                conn.execute(
                    f"INSERT OR REPLACE INTO {table} ({col_sql}) VALUES ({placeholders})",
                    values,
                )
        conn.commit()


class Repository:
    def __init__(self, db_path: Path) -> None:
        self.path = db_path
        self.lock = threading.RLock()

    def init(self) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS channels (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        owner_id INTEGER,
                        telegram_id INTEGER,
                        channel_url TEXT,
                        channel_title TEXT,
                        channel_username TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        channel_id INTEGER,
                        telegram_message_id INTEGER,
                        message_date TEXT,
                        message_url TEXT,
                        caption TEXT,
                        FOREIGN KEY(channel_id) REFERENCES channels(id)
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS media_items (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        message_id INTEGER,
                        file_name TEXT,
                        file_path TEXT,
                        mime_type TEXT,
                        media_kind TEXT,
                        duration_seconds INTEGER,
                        file_size_bytes INTEGER,
                        transcript_text TEXT,
                        transcript_status TEXT DEFAULT 'pending',
                        transcript_error TEXT,
                        FOREIGN KEY(message_id) REFERENCES messages(id)
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        media_item_id INTEGER,
                        chunk_index INTEGER,
                        text TEXT,
                        embedding BLOB,
                        start_char INTEGER,
                        end_char INTEGER,
                        FOREIGN KEY(media_item_id) REFERENCES media_items(id)
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS bot_users (
                        bot_user_id INTEGER PRIMARY KEY,
                        chat_id INTEGER,
                        username TEXT,
                        first_name TEXT,
                        phone TEXT,
                        api_id INTEGER,
                        api_hash TEXT,
                        session_string TEXT,
                        connected_at TEXT,
                        preferred_transcription_model TEXT DEFAULT 'gemini-2.5-flash-lite',
                        preferred_embedding_model TEXT DEFAULT 'text-embedding-004',
                        gemini_api_key TEXT,
                        vertex_project_id TEXT,
                        vertex_region TEXT,
                        vertex_index_id TEXT,
                        vertex_endpoint_id TEXT,
                        vertex_deployed_index_id TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS auth_flows (
                        bot_user_id INTEGER PRIMARY KEY,
                        chat_id INTEGER,
                        phone TEXT,
                        api_id INTEGER,
                        api_hash TEXT,
                        session_string TEXT,
                        phone_code_hash TEXT,
                        status TEXT,
                        vertex_project_id TEXT,
                        vertex_region TEXT,
                        vertex_index_id TEXT,
                        vertex_endpoint_id TEXT,
                        vertex_deployed_index_id TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS rules (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        owner_id INTEGER,
                        keyword TEXT,
                        tag TEXT,
                        created_at TEXT
                    )
                """)
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_rules_owner_kw_tag ON rules(owner_id, keyword, tag)"
                )
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS content_tags (
                        owner_id INTEGER,
                        media_item_id INTEGER,
                        tag TEXT,
                        PRIMARY KEY (owner_id, media_item_id, tag)
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS jobs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        owner_id INTEGER,
                        channel_url TEXT,
                        status TEXT,
                        total INTEGER,
                        processed INTEGER DEFAULT 0,
                        cursor INTEGER DEFAULT 0,
                        limit_count INTEGER,
                        error TEXT,
                        cancel_requested INTEGER DEFAULT 0,
                        created_at TEXT,
                        updated_at TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS collections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        owner_id INTEGER,
                        name TEXT,
                        created_at TEXT
                    )
                """)
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_collections_owner_name ON collections(owner_id, name)"
                )
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS collection_tags (
                        collection_id INTEGER,
                        tag TEXT,
                        PRIMARY KEY (collection_id, tag)
                    )
                """)
                # Live cache of incoming messages for the deleted-message watcher.
                # Only populated while a user has /watchdeleted ON.
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cached_messages (
                        owner_id INTEGER,
                        chat_id INTEGER,
                        message_id INTEGER,
                        chat_title TEXT,
                        sender TEXT,
                        text TEXT,
                        message_date TEXT,
                        PRIMARY KEY (owner_id, chat_id, message_id)
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS recovered_deleted (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        owner_id INTEGER,
                        chat_id INTEGER,
                        message_id INTEGER,
                        chat_title TEXT,
                        sender TEXT,
                        text TEXT,
                        message_date TEXT,
                        deleted_at TEXT
                    )
                """)
                self._ensure_channel_owner(conn)
                self._ensure_bot_user_columns(conn)
                self._ensure_rule_columns(conn)
                self._ensure_chunk_columns(conn)
                conn.commit()

    @staticmethod
    def _ensure_bot_user_columns(conn: sqlite3.Connection) -> None:
        """Add columns introduced after the initial bot_users schema (idempotent)."""
        cols = [row[1] for row in conn.execute("PRAGMA table_info(bot_users)").fetchall()]
        if "archive_chat_id" not in cols:
            conn.execute("ALTER TABLE bot_users ADD COLUMN archive_chat_id TEXT")
        if "ai_autotag" not in cols:
            conn.execute("ALTER TABLE bot_users ADD COLUMN ai_autotag INTEGER DEFAULT 0")
        if "watch_deleted" not in cols:
            conn.execute("ALTER TABLE bot_users ADD COLUMN watch_deleted INTEGER DEFAULT 0")

    @staticmethod
    def _ensure_chunk_columns(conn: sqlite3.Connection) -> None:
        """Record which embedding space each stored vector belongs to (idempotent).

        Without these, a corpus embedded by two different models is
        indistinguishable from one embedded by a single model: the cosine
        between vectors from two spaces is a plausible-looking number with no
        relationship to similarity, so nothing fails and recall just decays.
        With them, a stale vector is a query away from being found.
        """
        cols = [row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()]
        if "embed_model" not in cols:
            conn.execute("ALTER TABLE chunks ADD COLUMN embed_model TEXT")
        if "embed_dim" not in cols:
            conn.execute("ALTER TABLE chunks ADD COLUMN embed_dim INTEGER")

    @staticmethod
    def _ensure_rule_columns(conn: sqlite3.Connection) -> None:
        """Add the rule ``kind`` column (keyword vs ai) to older databases (idempotent)."""
        cols = [row[1] for row in conn.execute("PRAGMA table_info(rules)").fetchall()]
        if "kind" not in cols:
            conn.execute("ALTER TABLE rules ADD COLUMN kind TEXT DEFAULT 'keyword'")

    @staticmethod
    def _ensure_channel_owner(conn: sqlite3.Connection) -> None:
        """Migrate older databases to per-user channel ownership.

        Adds the ``owner_id`` column and replaces the global ``UNIQUE(channel_url)``
        constraint with a composite ``UNIQUE(owner_id, channel_url)`` index so that
        two users can independently ingest the same channel URL without sharing rows.
        Legacy rows (created before ownership existed) keep ``owner_id = NULL`` and are
        therefore invisible to per-user queries rather than leaking across users.
        """
        cols = [row[1] for row in conn.execute("PRAGMA table_info(channels)").fetchall()]
        if "owner_id" not in cols:
            conn.execute("ALTER TABLE channels RENAME TO channels_legacy")
            conn.execute("""
                CREATE TABLE channels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id INTEGER,
                    telegram_id INTEGER,
                    channel_url TEXT,
                    channel_title TEXT,
                    channel_username TEXT
                )
            """)
            conn.execute("""
                INSERT INTO channels (id, owner_id, telegram_id, channel_url, channel_title, channel_username)
                SELECT id, NULL, telegram_id, channel_url, channel_title, channel_username FROM channels_legacy
            """)
            conn.execute("DROP TABLE channels_legacy")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_channels_owner_url ON channels(owner_id, channel_url)"
        )

    def upsert_channel(self, *, owner_id: int, telegram_id: int, channel_url: str, title: str | None, username: str | None) -> int:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("""
                    INSERT INTO channels (owner_id, telegram_id, channel_url, channel_title, channel_username)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(owner_id, channel_url) DO UPDATE SET
                        telegram_id=excluded.telegram_id,
                        channel_title=excluded.channel_title,
                        channel_username=excluded.channel_username
                """, (owner_id, telegram_id, channel_url, title, username))
                res = conn.execute(
                    "SELECT id FROM channels WHERE owner_id = ? AND channel_url = ?",
                    (owner_id, channel_url),
                ).fetchone()
                return res[0]

    def create_or_get_message(self, *, channel_id: int, telegram_message_id: int, message_date: str | None, message_url: str | None, caption: str | None) -> int:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                res = conn.execute("SELECT id FROM messages WHERE channel_id = ? AND telegram_message_id = ?", (channel_id, telegram_message_id)).fetchone()
                if res:
                    conn.execute("UPDATE messages SET caption = ?, message_url = ? WHERE id = ?", (caption, message_url, res[0]))
                    return res[0]
                
                cursor = conn.execute("""
                    INSERT INTO messages (channel_id, telegram_message_id, message_date, message_url, caption)
                    VALUES (?, ?, ?, ?, ?)
                """, (channel_id, telegram_message_id, message_date, message_url, caption))
                return cursor.lastrowid

    def create_or_get_media(self, *, message_id: int, file_name: str | None, file_path: str, mime_type: str | None, media_kind: str, duration_seconds: int | None, file_size_bytes: int | None) -> int:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                res = conn.execute("SELECT id FROM media_items WHERE message_id = ?", (message_id,)).fetchone()
                if res:
                    return res[0]
                
                cursor = conn.execute("""
                    INSERT INTO media_items (message_id, file_name, file_path, mime_type, media_kind, duration_seconds, file_size_bytes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (message_id, file_name, file_path, mime_type, media_kind, duration_seconds, file_size_bytes))
                return cursor.lastrowid

    def replace_chunks(self, *, media_item_id: int, chunks: list[dict[str, Any]]) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("DELETE FROM chunks WHERE media_item_id = ?", (media_item_id,))
                for chunk in chunks:
                    vector = chunk.get("embedding")
                    embedding_blob = json.dumps(vector).encode('utf-8') if vector else None
                    conn.execute("""
                        INSERT INTO chunks (media_item_id, chunk_index, text, embedding, start_char, end_char,
                                            embed_model, embed_dim)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (media_item_id, chunk["chunk_index"], chunk["text"], embedding_blob,
                          chunk["start_char"], chunk["end_char"],
                          chunk.get("embed_model"), len(vector) if vector else None))
                conn.commit()

    def mark_media_transcribed(self, *, media_item_id: int, transcript_text: str) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("UPDATE media_items SET transcript_text = ?, transcript_status = 'done' WHERE id = ?", (transcript_text, media_item_id))
                conn.commit()

    def mark_media_failed(self, *, media_item_id: int, error: str) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute(
                    "UPDATE media_items SET transcript_status = 'error', transcript_error = ? WHERE id = ?",
                    (error, media_item_id),
                )
                conn.commit()

    def media_already_transcribed(self, media_item_id: int) -> bool:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                res = conn.execute("SELECT transcript_status FROM media_items WHERE id = ?", (media_item_id,)).fetchone()
                return res and res[0] == 'done'

    def keyword_candidates(self, *, owner_id: int, query: str, top_k: int, channel_url: str | None, tag: str | None = None) -> list[dict[str, Any]]:
        # Lexical search: match per-term (not the whole query as one substring) so
        # natural-language questions still hit, then rank by how many distinct terms
        # a chunk contains.
        terms = query_terms(query)
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                sql = """
                    SELECT c.text as chunk_text, m.message_url, ch.channel_title, ch.channel_url
                    FROM chunks c
                    JOIN media_items mi ON c.media_item_id = mi.id
                    JOIN messages m ON mi.message_id = m.id
                    JOIN channels ch ON m.channel_id = ch.id
                    WHERE ch.owner_id = ?
                """
                params: list[Any] = [owner_id]
                if terms:
                    sql += " AND (" + " OR ".join("c.text LIKE ?" for _ in terms) + ")"
                    params.extend(f"%{t}%" for t in terms)
                else:
                    # No usable terms (e.g. punctuation/very short): fall back to the
                    # whole query as a single substring.
                    sql += " AND c.text LIKE ?"
                    params.append(f"%{query}%")
                if channel_url:
                    sql += " AND ch.channel_url = ?"
                    params.append(channel_url)
                if tag:
                    sql += """ AND mi.id IN (
                        SELECT media_item_id FROM content_tags WHERE owner_id = ? AND tag = ?
                    )"""
                    params.extend([owner_id, tag])

                # Over-fetch, then rank in Python by distinct-term hits and trim.
                fetch_limit = max(top_k * 10, 50)
                rows = [dict(r) for r in conn.execute(sql + " LIMIT ?", params + [fetch_limit]).fetchall()]

        if terms and len(terms) > 1:
            def _term_hits(row: dict[str, Any]) -> int:
                text = (row.get("chunk_text") or "").lower()
                return sum(1 for t in terms if t in text)
            rows.sort(key=_term_hits, reverse=True)
        return rows[:top_k]

    def embedding_candidates(self, *, owner_id: int, channel_url: str | None, tag: str | None = None) -> list[dict[str, Any]]:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                sql = """
                    SELECT c.id as chunk_id, c.text as chunk_text, c.embedding as embedding_json,
                           mi.media_kind, mi.file_name, m.message_url, m.caption,
                           ch.channel_title, ch.channel_url
                    FROM chunks c
                    JOIN media_items mi ON c.media_item_id = mi.id
                    JOIN messages m ON mi.message_id = m.id
                    JOIN channels ch ON m.channel_id = ch.id
                    WHERE c.embedding IS NOT NULL AND ch.owner_id = ?
                """
                params: list[Any] = [owner_id]
                if channel_url:
                    sql += " AND ch.channel_url = ?"
                    params.append(channel_url)
                if tag:
                    sql += """ AND mi.id IN (
                        SELECT media_item_id FROM content_tags WHERE owner_id = ? AND tag = ?
                    )"""
                    params.extend([owner_id, tag])

                rows = conn.execute(sql, params).fetchall()
                results = []
                for r in rows:
                    d = dict(r)
                    if d["embedding_json"]:
                        d["embedding_json"] = d["embedding_json"].decode('utf-8')
                    results.append(d)
                return results

    def list_channels(self, *, owner_id: int) -> list[dict[str, Any]]:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                return [dict(r) for r in conn.execute(
                    "SELECT * FROM channels WHERE owner_id = ?", (owner_id,)
                ).fetchall()]

    def delete_channel_data(self, *, owner_id: int, channel_url: str) -> bool:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                res = conn.execute(
                    "SELECT id FROM channels WHERE owner_id = ? AND channel_url = ?",
                    (owner_id, channel_url),
                ).fetchone()
                if not res:
                    return False
                cid = res[0]
                conn.execute("DELETE FROM chunks WHERE media_item_id IN (SELECT id FROM media_items WHERE message_id IN (SELECT id FROM messages WHERE channel_id = ?))", (cid,))
                conn.execute("DELETE FROM media_items WHERE message_id IN (SELECT id FROM messages WHERE channel_id = ?)", (cid,))
                conn.execute("DELETE FROM messages WHERE channel_id = ?", (cid,))
                conn.execute("DELETE FROM channels WHERE id = ?", (cid,))
                conn.commit()
                return True

    def get_chunk_by_media_and_index(self, *, owner_id: int, media_item_id: int, chunk_index: int) -> dict[str, Any] | None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("""
                    SELECT c.text as chunk_text, mi.media_kind, mi.file_name,
                           m.message_url, m.caption, ch.channel_title, ch.channel_url
                    FROM chunks c
                    JOIN media_items mi ON c.media_item_id = mi.id
                    JOIN messages m ON mi.message_id = m.id
                    JOIN channels ch ON m.channel_id = ch.id
                    WHERE c.media_item_id = ? AND c.chunk_index = ? AND ch.owner_id = ?
                """, (media_item_id, chunk_index, owner_id)).fetchone()
                return dict(row) if row else None

    def upsert_bot_user(
self, *, bot_user_id: int, chat_id: int, username: str | None, first_name: str | None) -> dict[str, Any]:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("""
                    INSERT INTO bot_users (bot_user_id, chat_id, username, first_name)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(bot_user_id) DO UPDATE SET
                        chat_id=excluded.chat_id, username=excluded.username, first_name=excluded.first_name
                """, (bot_user_id, chat_id, username, first_name))
                conn.commit()
                conn.row_factory = sqlite3.Row
                return _decrypt_row(dict(conn.execute("SELECT * FROM bot_users WHERE bot_user_id = ?", (bot_user_id,)).fetchone()), _BOT_USER_SECRETS)

    def get_bot_user(self, *, bot_user_id: int) -> dict[str, Any] | None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                res = conn.execute("SELECT * FROM bot_users WHERE bot_user_id = ?", (bot_user_id,)).fetchone()
                return _decrypt_row(dict(res), _BOT_USER_SECRETS) if res else None

    def set_ai_autotag(self, *, bot_user_id: int, enabled: bool) -> None:
        """Toggle whether AI rules run automatically on the user's new forwards."""
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute(
                    "UPDATE bot_users SET ai_autotag = ? WHERE bot_user_id = ?",
                    (1 if enabled else 0, bot_user_id),
                )

    def set_watch_deleted(self, *, bot_user_id: int, enabled: bool) -> None:
        """Toggle the always-on deleted-message watcher for a user."""
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute(
                    "UPDATE bot_users SET watch_deleted = ? WHERE bot_user_id = ?",
                    (1 if enabled else 0, bot_user_id),
                )

    def list_watch_deleted_owners(self) -> list[dict[str, Any]]:
        """Connected users who opted into the deleted-message watcher (decrypted)."""
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM bot_users WHERE watch_deleted = 1 AND session_string IS NOT NULL AND session_string != ''"
                ).fetchall()
        owners = []
        for r in rows:
            d = _decrypt_row(dict(r), _BOT_USER_SECRETS)
            if d and d.get("session_string"):
                owners.append(d)
        return owners

    def cache_message(
        self,
        *,
        owner_id: int,
        chat_id: int | None,
        message_id: int,
        chat_title: str | None,
        sender: str | None,
        text: str | None,
        message_date: str | None,
    ) -> None:
        """Store one incoming message so it can be recovered if later deleted."""
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cached_messages
                        (owner_id, chat_id, message_id, chat_title, sender, text, message_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (owner_id, chat_id, message_id, chat_title, sender, text, message_date),
                )

    def take_cached(self, *, owner_id: int, message_ids: list[int], chat_id: int | None = None) -> list[dict[str, Any]]:
        """Fetch + remove cached messages matching ``message_ids`` (optionally a chat).

        Returned rows are the cached copies of messages that were just deleted.
        """
        if not message_ids:
            return []
        placeholders = ", ".join("?" for _ in message_ids)
        params: list[Any] = [owner_id, *message_ids]
        where = f"owner_id = ? AND message_id IN ({placeholders})"
        if chat_id is not None:
            where += " AND chat_id = ?"
            params.append(chat_id)
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                rows = [dict(r) for r in conn.execute(f"SELECT * FROM cached_messages WHERE {where}", params).fetchall()]
                if rows:
                    conn.execute(f"DELETE FROM cached_messages WHERE {where}", params)
                return rows

    def add_recovered(self, *, owner_id: int, rows: list[dict[str, Any]], deleted_at: str) -> int:
        """Persist recovered (deleted) messages. Returns how many were stored."""
        if not rows:
            return 0
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.executemany(
                    """
                    INSERT INTO recovered_deleted
                        (owner_id, chat_id, message_id, chat_title, sender, text, message_date, deleted_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            owner_id, r.get("chat_id"), r.get("message_id"), r.get("chat_title"),
                            r.get("sender"), r.get("text"), r.get("message_date"), deleted_at,
                        )
                        for r in rows
                    ],
                )
        return len(rows)

    def list_recovered(self, *, owner_id: int, limit: int = 20) -> list[dict[str, Any]]:
        """Most-recently recovered deleted messages for a user (newest first)."""
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM recovered_deleted WHERE owner_id = ? ORDER BY id DESC LIMIT ?",
                    (owner_id, limit),
                ).fetchall()
                return [dict(r) for r in rows]

    def set_archive_chat(self, *, bot_user_id: int, archive_chat_id: str | None) -> None:
        """Set (or clear, when ``None``) the channel/chat that tagged forwards are auto-forwarded to."""
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute(
                    "UPDATE bot_users SET archive_chat_id = ? WHERE bot_user_id = ?",
                    (archive_chat_id, bot_user_id),
                )

    def disconnect_bot_user(self, *, bot_user_id: int) -> bool:
        """Remove a user's session and credentials. Returns True if a session existed."""
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                res = conn.execute(
                    "SELECT session_string FROM bot_users WHERE bot_user_id = ?",
                    (bot_user_id,),
                ).fetchone()
                had_session = bool(res and res[0])
                conn.execute(
                    """
                    UPDATE bot_users SET
                        session_string = NULL, api_id = NULL, api_hash = NULL,
                        phone = NULL, connected_at = NULL, gemini_api_key = NULL,
                        vertex_project_id = NULL, vertex_region = NULL, vertex_index_id = NULL,
                        vertex_endpoint_id = NULL, vertex_deployed_index_id = NULL
                    WHERE bot_user_id = ?
                    """,
                    (bot_user_id,),
                )
                conn.commit()
                return had_session

    def save_bot_user_phone(self, *, bot_user_id: int, phone: str) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("UPDATE bot_users SET phone = ? WHERE bot_user_id = ?", (phone, bot_user_id))

    def save_bot_user_session(self, *, bot_user_id: int, phone: str, api_id: int | None, api_hash: str | None, session_string: str, connected_at: str, 
                              v_project: str | None = None, v_region: str | None = None, v_index: str | None = None, v_endpoint: str | None = None, v_deployed: str | None = None) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("""
                    UPDATE bot_users SET phone=?, api_id=?, api_hash=?, session_string=?, connected_at=?,
                                       vertex_project_id=?, vertex_region=?, vertex_index_id=?, vertex_endpoint_id=?, vertex_deployed_index_id=?
                    WHERE bot_user_id=?
                """, (phone, api_id, _encrypt(api_hash), _encrypt(session_string), connected_at, v_project, v_region, v_index, v_endpoint, v_deployed, bot_user_id))

    def update_user_gemini_key(self, *, bot_user_id: int, api_key: str) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("UPDATE bot_users SET gemini_api_key = ? WHERE bot_user_id = ?", (_encrypt(api_key), bot_user_id))

    def update_user_models(self, bot_user_id: int, transcription_model: str, embedding_model: str) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("UPDATE bot_users SET preferred_transcription_model=?, preferred_embedding_model=? WHERE bot_user_id=?", (transcription_model, embedding_model, bot_user_id))

    def upsert_auth_flow(self, *, bot_user_id: int, chat_id: int, phone: str, api_id: int | None, api_hash: str | None, session_string: str, phone_code_hash: str, status: str,
                        v_project: str | None = None, v_region: str | None = None, v_index: str | None = None, v_endpoint: str | None = None, v_deployed: str | None = None) -> dict[str, Any]:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("""
                    INSERT INTO auth_flows (bot_user_id, chat_id, phone, api_id, api_hash, session_string, phone_code_hash, status,
                                          vertex_project_id, vertex_region, vertex_index_id, vertex_endpoint_id, vertex_deployed_index_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(bot_user_id) DO UPDATE SET
                        chat_id=excluded.chat_id, phone=excluded.phone, api_id=excluded.api_id,
                        api_hash=excluded.api_hash, session_string=excluded.session_string,
                        phone_code_hash=excluded.phone_code_hash, status=excluded.status,
                        vertex_project_id=excluded.vertex_project_id, vertex_region=excluded.vertex_region,
                        vertex_index_id=excluded.vertex_index_id, vertex_endpoint_id=excluded.vertex_endpoint_id,
                        vertex_deployed_index_id=excluded.vertex_deployed_index_id
                """, (bot_user_id, chat_id, phone, api_id, _encrypt(api_hash), _encrypt(session_string), _encrypt(phone_code_hash), status, v_project, v_region, v_index, v_endpoint, v_deployed))
                conn.commit()
                conn.row_factory = sqlite3.Row
                return _decrypt_row(dict(conn.execute("SELECT * FROM auth_flows WHERE bot_user_id = ?", (bot_user_id,)).fetchone()), _AUTH_FLOW_SECRETS)

    def update_auth_flow_status(self, *, bot_user_id: int, status: str) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("UPDATE auth_flows SET status = ? WHERE bot_user_id = ?", (status, bot_user_id))

    def get_auth_flow(self, *, bot_user_id: int) -> dict[str, Any] | None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                res = conn.execute("SELECT * FROM auth_flows WHERE bot_user_id = ?", (bot_user_id,)).fetchone()
                return _decrypt_row(dict(res), _AUTH_FLOW_SECRETS) if res else None

    def clear_auth_flow(self, *, bot_user_id: int) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("DELETE FROM auth_flows WHERE bot_user_id = ?", (bot_user_id,))

    # --- Rules & tags ------------------------------------------------------

    def add_rule(self, *, owner_id: int, keyword: str, tag: str, kind: str = "keyword", created_at: str | None = None) -> int:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO rules (owner_id, keyword, tag, kind, created_at) VALUES (?, ?, ?, ?, ?)",
                    (owner_id, keyword, tag, kind, created_at),
                )
                conn.commit()
                res = conn.execute(
                    "SELECT id FROM rules WHERE owner_id = ? AND keyword = ? AND tag = ?",
                    (owner_id, keyword, tag),
                ).fetchone()
                return res[0] if res else 0

    def list_rules(self, *, owner_id: int) -> list[dict[str, Any]]:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                return [dict(r) for r in conn.execute(
                    "SELECT id, keyword, tag, COALESCE(kind, 'keyword') AS kind FROM rules WHERE owner_id = ? ORDER BY id",
                    (owner_id,),
                ).fetchall()]

    def remove_rule(self, *, owner_id: int, rule_id: int) -> bool:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                cur = conn.execute(
                    "DELETE FROM rules WHERE owner_id = ? AND id = ?", (owner_id, rule_id)
                )
                conn.commit()
                return cur.rowcount > 0

    def tag_media(self, *, owner_id: int, media_item_id: int, tags: set[str] | list[str]) -> None:
        if not tags:
            return
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO content_tags (owner_id, media_item_id, tag) VALUES (?, ?, ?)",
                    [(owner_id, media_item_id, tag) for tag in tags],
                )
                conn.commit()

    def list_tags(self, *, owner_id: int) -> list[dict[str, Any]]:
        """Tags with the number of distinct content items each one covers."""
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                return [dict(r) for r in conn.execute(
                    """
                    SELECT tag, COUNT(DISTINCT media_item_id) AS count
                    FROM content_tags WHERE owner_id = ?
                    GROUP BY tag ORDER BY count DESC, tag
                    """,
                    (owner_id,),
                ).fetchall()]

    def media_ids_for_tag(self, *, owner_id: int, tag: str) -> set[int]:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                return {
                    row[0]
                    for row in conn.execute(
                        "SELECT media_item_id FROM content_tags WHERE owner_id = ? AND tag = ?",
                        (owner_id, tag),
                    ).fetchall()
                }

    def clear_tags(self, *, owner_id: int) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("DELETE FROM content_tags WHERE owner_id = ?", (owner_id,))
                conn.commit()

    def rename_tag(self, *, owner_id: int, old_tag: str, new_tag: str) -> int:
        """Rename (or merge into an existing) tag across the owner's content. Returns rows moved.

        Items already carrying ``new_tag`` collapse cleanly (INSERT OR IGNORE then DELETE),
        so renaming onto an existing tag merges the two.
        """
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                affected = conn.execute(
                    "SELECT COUNT(*) FROM content_tags WHERE owner_id = ? AND tag = ?", (owner_id, old_tag)
                ).fetchone()[0]
                if affected:
                    conn.execute(
                        "INSERT OR IGNORE INTO content_tags (owner_id, media_item_id, tag) "
                        "SELECT owner_id, media_item_id, ? FROM content_tags WHERE owner_id = ? AND tag = ?",
                        (new_tag, owner_id, old_tag),
                    )
                    conn.execute("DELETE FROM content_tags WHERE owner_id = ? AND tag = ?", (owner_id, old_tag))
                    conn.commit()
                return affected

    def delete_tag(self, *, owner_id: int, tag: str) -> int:
        """Remove a tag from all of the owner's content. Returns rows removed."""
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                cur = conn.execute("DELETE FROM content_tags WHERE owner_id = ? AND tag = ?", (owner_id, tag))
                conn.commit()
                return cur.rowcount

    # --- Collections (named bundles of tags) -------------------------------

    def create_collection(self, *, owner_id: int, name: str, created_at: str | None = None) -> int:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO collections (owner_id, name, created_at) VALUES (?, ?, ?)",
                    (owner_id, name, created_at),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT id FROM collections WHERE owner_id = ? AND name = ?", (owner_id, name)
                ).fetchone()
                return row[0] if row else 0

    def _collection_id(self, conn: sqlite3.Connection, owner_id: int, name: str) -> int | None:
        row = conn.execute(
            "SELECT id FROM collections WHERE owner_id = ? AND name = ?", (owner_id, name)
        ).fetchone()
        return row[0] if row else None

    def add_collection_tag(self, *, owner_id: int, name: str, tag: str) -> bool:
        """Add a tag to a collection. Returns False if the collection doesn't exist."""
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                cid = self._collection_id(conn, owner_id, name)
                if cid is None:
                    return False
                conn.execute(
                    "INSERT OR IGNORE INTO collection_tags (collection_id, tag) VALUES (?, ?)", (cid, tag)
                )
                conn.commit()
                return True

    def list_collections(self, *, owner_id: int) -> list[dict[str, Any]]:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                cols = conn.execute(
                    "SELECT id, name FROM collections WHERE owner_id = ? ORDER BY name", (owner_id,)
                ).fetchall()
                out = []
                for c in cols:
                    tags = [r[0] for r in conn.execute(
                        "SELECT tag FROM collection_tags WHERE collection_id = ? ORDER BY tag", (c["id"],)
                    ).fetchall()]
                    out.append({"id": c["id"], "name": c["name"], "tags": tags})
                return out

    def collection_tags(self, *, owner_id: int, name: str) -> list[str] | None:
        """Tags in a collection, or None if the collection doesn't exist."""
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                cid = self._collection_id(conn, owner_id, name)
                if cid is None:
                    return None
                return [r[0] for r in conn.execute(
                    "SELECT tag FROM collection_tags WHERE collection_id = ? ORDER BY tag", (cid,)
                ).fetchall()]

    def remove_collection(self, *, owner_id: int, name: str) -> bool:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                cid = self._collection_id(conn, owner_id, name)
                if cid is None:
                    return False
                conn.execute("DELETE FROM collection_tags WHERE collection_id = ?", (cid,))
                conn.execute("DELETE FROM collections WHERE id = ?", (cid,))
                conn.commit()
                return True

    def items_for_tags(self, *, owner_id: int, tags: list[str], limit: int = 200) -> list[dict[str, Any]]:
        """Distinct content items carrying ANY of ``tags`` (text + source), newest first."""
        if not tags:
            return []
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                placeholders = ",".join("?" for _ in tags)
                rows = conn.execute(
                    f"""
                    SELECT m.message_date, mi.transcript_text AS text, ch.channel_title, ch.channel_url, m.message_url
                    FROM media_items mi
                    JOIN messages m ON mi.message_id = m.id
                    JOIN channels ch ON m.channel_id = ch.id
                    WHERE ch.owner_id = ? AND mi.transcript_text IS NOT NULL AND mi.transcript_text != ''
                      AND mi.id IN (
                        SELECT media_item_id FROM content_tags WHERE owner_id = ? AND tag IN ({placeholders})
                      )
                    ORDER BY m.message_date DESC LIMIT ?
                    """,
                    (owner_id, owner_id, *tags, limit),
                ).fetchall()
                return [dict(r) for r in rows]

    def media_texts(self, *, owner_id: int) -> list[dict[str, Any]]:
        """Every stored media item for the owner with its transcript/text, for re-tagging."""
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                return [dict(r) for r in conn.execute(
                    """
                    SELECT mi.id AS media_item_id, mi.transcript_text AS text
                    FROM media_items mi
                    JOIN messages m ON mi.message_id = m.id
                    JOIN channels ch ON m.channel_id = ch.id
                    WHERE ch.owner_id = ? AND mi.transcript_text IS NOT NULL
                    """,
                    (owner_id,),
                ).fetchall()]

    def get_media_item(self, *, owner_id: int, media_item_id: int) -> dict[str, Any] | None:
        """Full stored text of a single content item (owner-scoped)."""
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    """
                    SELECT mi.transcript_text AS text, mi.media_kind, mi.file_name,
                           m.message_url, m.caption, ch.channel_title, ch.channel_url
                    FROM media_items mi
                    JOIN messages m ON mi.message_id = m.id
                    JOIN channels ch ON m.channel_id = ch.id
                    WHERE mi.id = ? AND ch.owner_id = ?
                    """,
                    (media_item_id, owner_id),
                ).fetchone()
                return dict(row) if row else None

    def summary_items(self, *, owner_id: int, channel_url: str | None = None, tag: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        """One row per stored content item (with its text and source), for summarization.
        Optionally scoped to a single source (``channel_url``) and/or ``tag``."""
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                sql = """
                    SELECT mi.transcript_text AS text, ch.channel_title, ch.channel_url, m.message_url
                    FROM media_items mi
                    JOIN messages m ON mi.message_id = m.id
                    JOIN channels ch ON m.channel_id = ch.id
                    WHERE ch.owner_id = ? AND mi.transcript_text IS NOT NULL AND mi.transcript_text != ''
                """
                params: list[Any] = [owner_id]
                if channel_url:
                    sql += " AND ch.channel_url = ?"
                    params.append(channel_url)
                if tag:
                    sql += " AND mi.id IN (SELECT media_item_id FROM content_tags WHERE owner_id = ? AND tag = ?)"
                    params.extend([owner_id, tag])
                sql += " ORDER BY m.id LIMIT ?"
                params.append(limit)
                return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def archive_stats(self, *, owner_id: int) -> dict[str, Any]:
        """Aggregate counts for the owner's archive: items, sources, tags, kinds, date range."""
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                base = (
                    "FROM media_items mi JOIN messages m ON mi.message_id = m.id "
                    "JOIN channels ch ON m.channel_id = ch.id "
                    "WHERE ch.owner_id = ? AND mi.transcript_text IS NOT NULL AND mi.transcript_text != ''"
                )
                total, first_date, last_date = conn.execute(
                    f"SELECT COUNT(*), MIN(m.message_date), MAX(m.message_date) {base}", (owner_id,)
                ).fetchone()
                by_kind = {
                    kind: count
                    for kind, count in conn.execute(
                        f"SELECT mi.media_kind, COUNT(*) {base} GROUP BY mi.media_kind ORDER BY COUNT(*) DESC", (owner_id,)
                    ).fetchall()
                }
                sources = conn.execute(
                    "SELECT COUNT(*) FROM channels WHERE owner_id = ?", (owner_id,)
                ).fetchone()[0]
                tags = conn.execute(
                    "SELECT COUNT(DISTINCT tag) FROM content_tags WHERE owner_id = ?", (owner_id,)
                ).fetchone()[0]
                return {
                    "items": total or 0,
                    "sources": sources or 0,
                    "tags": tags or 0,
                    "by_kind": by_kind,
                    "first_date": first_date,
                    "last_date": last_date,
                }

    def timeline_items(self, *, owner_id: int, channel_url: str | None = None, tag: str | None = None, limit: int = 2000) -> list[dict[str, Any]]:
        """Dated content items (date + text + source), newest first, for the timeline view.
        Optionally scoped to a single source (``channel_url``) and/or ``tag``."""
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                sql = """
                    SELECT m.message_date, mi.transcript_text AS text, ch.channel_title, ch.channel_url, m.message_url
                    FROM media_items mi
                    JOIN messages m ON mi.message_id = m.id
                    JOIN channels ch ON m.channel_id = ch.id
                    WHERE ch.owner_id = ? AND m.message_date IS NOT NULL AND m.message_date != ''
                """
                params: list[Any] = [owner_id]
                if channel_url:
                    sql += " AND ch.channel_url = ?"
                    params.append(channel_url)
                if tag:
                    sql += " AND mi.id IN (SELECT media_item_id FROM content_tags WHERE owner_id = ? AND tag = ?)"
                    params.extend([owner_id, tag])
                sql += " ORDER BY m.message_date DESC LIMIT ?"
                params.append(limit)
                return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def recent_items(self, *, owner_id: int, since_date: str, limit: int = 200) -> list[dict[str, Any]]:
        """Content items with ``message_date >= since_date`` (ISO), newest first, for digests."""
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT m.message_date, mi.transcript_text AS text, ch.channel_title, ch.channel_url, m.message_url
                    FROM media_items mi
                    JOIN messages m ON mi.message_id = m.id
                    JOIN channels ch ON m.channel_id = ch.id
                    WHERE ch.owner_id = ? AND mi.transcript_text IS NOT NULL AND mi.transcript_text != ''
                      AND m.message_date IS NOT NULL AND m.message_date >= ?
                    ORDER BY m.message_date DESC LIMIT ?
                    """,
                    (owner_id, since_date, limit),
                ).fetchall()
                return [dict(r) for r in rows]

    def chunks_with_embeddings(self, *, owner_id: int, channel_url: str | None = None, tag: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        """Chunks that have a stored embedding (with source), for offline topic clustering."""
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                sql = """
                    SELECT c.text AS text, c.embedding AS embedding, ch.channel_title, ch.channel_url, m.message_url
                    FROM chunks c
                    JOIN media_items mi ON c.media_item_id = mi.id
                    JOIN messages m ON mi.message_id = m.id
                    JOIN channels ch ON m.channel_id = ch.id
                    WHERE ch.owner_id = ? AND c.embedding IS NOT NULL
                """
                params: list[Any] = [owner_id]
                if channel_url:
                    sql += " AND ch.channel_url = ?"
                    params.append(channel_url)
                if tag:
                    sql += " AND mi.id IN (SELECT media_item_id FROM content_tags WHERE owner_id = ? AND tag = ?)"
                    params.extend([owner_id, tag])
                sql += " ORDER BY c.id LIMIT ?"
                params.append(limit)
                rows = conn.execute(sql, params).fetchall()
                items: list[dict[str, Any]] = []
                for r in rows:
                    d = dict(r)
                    raw = d.pop("embedding")
                    try:
                        d["embedding"] = json.loads(raw.decode("utf-8")) if raw else None
                    except (ValueError, AttributeError):
                        d["embedding"] = None
                    if d["embedding"]:
                        items.append(d)
                return items

    # --- Import jobs (queue / progress / resume) ---------------------------

    def create_job(self, *, owner_id: int, channel_url: str, limit: int | None, created_at: str | None = None) -> int:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                cur = conn.execute(
                    """
                    INSERT INTO jobs (owner_id, channel_url, status, processed, cursor, limit_count, cancel_requested, created_at, updated_at)
                    VALUES (?, ?, 'queued', 0, 0, ?, 0, ?, ?)
                    """,
                    (owner_id, channel_url, limit, created_at, created_at),
                )
                conn.commit()
                return cur.lastrowid

    def get_job(self, *, job_id: int, owner_id: int | None = None) -> dict[str, Any] | None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                if owner_id is None:
                    res = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                else:
                    res = conn.execute("SELECT * FROM jobs WHERE id = ? AND owner_id = ?", (job_id, owner_id)).fetchone()
                return dict(res) if res else None

    def list_jobs(self, *, owner_id: int, limit: int = 10) -> list[dict[str, Any]]:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                return [dict(r) for r in conn.execute(
                    "SELECT * FROM jobs WHERE owner_id = ? ORDER BY id DESC LIMIT ?", (owner_id, limit)
                ).fetchall()]

    def claim_next_queued_job(self, *, updated_at: str | None = None) -> dict[str, Any] | None:
        """Atomically pick the oldest queued job and mark it running."""
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1"
                ).fetchone()
                if not row:
                    return None
                conn.execute(
                    "UPDATE jobs SET status = 'running', updated_at = ? WHERE id = ?",
                    (updated_at, row["id"]),
                )
                conn.commit()
                job = dict(row)
                job["status"] = "running"
                return job

    def update_job_progress(self, *, job_id: int, processed: int, total: int | None, cursor: int | None, updated_at: str | None = None) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute(
                    "UPDATE jobs SET processed = ?, total = ?, cursor = ?, updated_at = ? WHERE id = ?",
                    (processed, total, cursor, updated_at, job_id),
                )
                conn.commit()

    def finish_job(self, *, job_id: int, status: str, error: str | None = None, updated_at: str | None = None) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute(
                    "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                    (status, error, updated_at, job_id),
                )
                conn.commit()

    def request_job_cancel(self, *, owner_id: int, job_id: int) -> bool:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                cur = conn.execute(
                    "UPDATE jobs SET cancel_requested = 1 WHERE id = ? AND owner_id = ? AND status IN ('queued', 'running')",
                    (job_id, owner_id),
                )
                conn.commit()
                return cur.rowcount > 0

    def is_cancel_requested(self, *, job_id: int) -> bool:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                res = conn.execute("SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)).fetchone()
                return bool(res and res[0])

    def requeue_running_jobs(self, *, updated_at: str | None = None) -> int:
        """Reset jobs left 'running' by a crashed worker back to 'queued' so they resume."""
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                cur = conn.execute(
                    "UPDATE jobs SET status = 'queued', updated_at = ? WHERE status = 'running'",
                    (updated_at,),
                )
                conn.commit()
                return cur.rowcount
