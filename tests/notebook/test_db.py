import sqlite3
from pathlib import Path

from telegram_notebook.db import Repository


def _repo(tmp_path: Path) -> Repository:
    repo = Repository(tmp_path / "store.db")
    repo.init()
    return repo


def test_channel_message_media_chunk_flow_and_search(tmp_path):
    repo = _repo(tmp_path)
    cid = repo.upsert_channel(owner_id=1, telegram_id=1, channel_url="https://t.me/c", title="C", username="c")
    # upsert is idempotent on (owner_id, channel_url)
    assert repo.upsert_channel(owner_id=1, telegram_id=1, channel_url="https://t.me/c", title="C2", username="c") == cid

    mid = repo.create_or_get_message(
        channel_id=cid, telegram_message_id=10, message_date=None,
        message_url="https://t.me/c/10", caption="cap",
    )
    assert repo.create_or_get_message(
        channel_id=cid, telegram_message_id=10, message_date=None,
        message_url="https://t.me/c/10", caption="cap",
    ) == mid

    media_id = repo.create_or_get_media(
        message_id=mid, file_name="text", file_path="", mime_type="text/plain",
        media_kind="text", duration_seconds=None, file_size_bytes=3,
    )
    repo.replace_chunks(media_item_id=media_id, chunks=[
        {"chunk_index": 0, "text": "hello world", "embedding": None, "start_char": 0, "end_char": 11},
    ])

    rows = repo.keyword_candidates(owner_id=1, query="hello", top_k=5, channel_url=None)
    assert len(rows) == 1
    assert rows[0]["chunk_text"] == "hello world"
    assert rows[0]["channel_url"] == "https://t.me/c"

    chunk = repo.get_chunk_by_media_and_index(owner_id=1, media_item_id=media_id, chunk_index=0)
    assert chunk and chunk["chunk_text"] == "hello world"

    assert len(repo.list_channels(owner_id=1)) == 1
    assert repo.delete_channel_data(owner_id=1, channel_url="https://t.me/c") is True
    assert repo.list_channels(owner_id=1) == []
    assert repo.delete_channel_data(owner_id=1, channel_url="https://t.me/c") is False


def test_keyword_candidates_filters_by_channel(tmp_path):
    repo = _repo(tmp_path)
    for url in ("https://t.me/a", "https://t.me/b"):
        cid = repo.upsert_channel(owner_id=1, telegram_id=hash(url) % 1000, channel_url=url, title=url, username=None)
        mid = repo.create_or_get_message(channel_id=cid, telegram_message_id=1, message_date=None, message_url=url + "/1", caption=None)
        media_id = repo.create_or_get_media(message_id=mid, file_name="t", file_path="", mime_type="text/plain", media_kind="text", duration_seconds=None, file_size_bytes=1)
        repo.replace_chunks(media_item_id=media_id, chunks=[{"chunk_index": 0, "text": "shared keyword", "embedding": None, "start_char": 0, "end_char": 14}])

    assert len(repo.keyword_candidates(owner_id=1, query="keyword", top_k=10, channel_url=None)) == 2
    only_a = repo.keyword_candidates(owner_id=1, query="keyword", top_k=10, channel_url="https://t.me/a")
    assert len(only_a) == 1 and only_a[0]["channel_url"] == "https://t.me/a"


def test_data_is_isolated_per_owner(tmp_path):
    """Two users ingesting the same channel URL must not see each other's data."""
    repo = _repo(tmp_path)

    def _seed(owner_id: int, text: str) -> None:
        cid = repo.upsert_channel(owner_id=owner_id, telegram_id=1, channel_url="https://t.me/shared", title="Shared", username=None)
        mid = repo.create_or_get_message(channel_id=cid, telegram_message_id=1, message_date=None, message_url="https://t.me/shared/1", caption=None)
        media_id = repo.create_or_get_media(message_id=mid, file_name="t", file_path="", mime_type="text/plain", media_kind="text", duration_seconds=None, file_size_bytes=1)
        repo.replace_chunks(media_item_id=media_id, chunks=[{"chunk_index": 0, "text": text, "embedding": None, "start_char": 0, "end_char": len(text)}])

    _seed(1, "secret of user one")
    _seed(2, "secret of user two")

    # Same URL, but each owner gets its own channel row.
    assert len(repo.list_channels(owner_id=1)) == 1
    assert len(repo.list_channels(owner_id=2)) == 1

    rows_one = repo.keyword_candidates(owner_id=1, query="secret", top_k=10, channel_url=None)
    assert len(rows_one) == 1 and rows_one[0]["chunk_text"] == "secret of user one"

    rows_two = repo.keyword_candidates(owner_id=2, query="secret", top_k=10, channel_url=None)
    assert len(rows_two) == 1 and rows_two[0]["chunk_text"] == "secret of user two"

    # A third user with no data sees nothing.
    assert repo.keyword_candidates(owner_id=3, query="secret", top_k=10, channel_url=None) == []

    # Deleting one owner's copy leaves the other intact.
    assert repo.delete_channel_data(owner_id=1, channel_url="https://t.me/shared") is True
    assert repo.list_channels(owner_id=1) == []
    assert len(repo.list_channels(owner_id=2)) == 1


def test_migrates_legacy_channels_table_without_owner_id(tmp_path):
    """An older DB whose channels table has no owner_id should upgrade cleanly."""
    db_path = tmp_path / "store.db"
    # Recreate the pre-ownership schema and seed a legacy row.
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE channels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_id INTEGER,
                channel_url TEXT UNIQUE,
                channel_title TEXT,
                channel_username TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO channels (telegram_id, channel_url, channel_title) VALUES (1, 'https://t.me/legacy', 'Legacy')"
        )
        conn.commit()

    repo = Repository(db_path)
    repo.init()  # runs the migration

    # owner_id column now exists; legacy row has NULL owner so it is invisible to users.
    cols = [r[1] for r in sqlite3.connect(db_path).execute("PRAGMA table_info(channels)").fetchall()]
    assert "owner_id" in cols
    assert repo.list_channels(owner_id=1) == []

    # New per-user ingest of the same URL works under the composite unique key.
    cid = repo.upsert_channel(owner_id=1, telegram_id=1, channel_url="https://t.me/legacy", title="Mine", username=None)
    assert repo.upsert_channel(owner_id=1, telegram_id=1, channel_url="https://t.me/legacy", title="Mine", username=None) == cid
    assert len(repo.list_channels(owner_id=1)) == 1


def test_secrets_are_encrypted_at_rest(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRETS_KEY", "unit-test-key")
    repo = _repo(tmp_path)
    repo.upsert_bot_user(bot_user_id=9, chat_id=9, username="u", first_name="f")
    repo.save_bot_user_session(bot_user_id=9, phone="+100", api_id=1, api_hash="HASH", session_string="SESS", connected_at="now")
    repo.update_user_gemini_key(bot_user_id=9, api_key="GKEY")

    # Raw storage is ciphertext, not the plaintext secret.
    with sqlite3.connect(tmp_path / "store.db") as conn:
        row = conn.execute(
            "SELECT api_hash, session_string, gemini_api_key, phone FROM bot_users WHERE bot_user_id = 9"
        ).fetchone()
    assert row[0].startswith("enc::") and "HASH" not in row[0]
    assert row[1].startswith("enc::") and "SESS" not in row[1]
    assert row[2].startswith("enc::") and "GKEY" not in row[2]
    assert row[3] == "+100"  # phone is not a credential; stored as-is

    # The repository transparently decrypts on read.
    user = repo.get_bot_user(bot_user_id=9)
    assert user["api_hash"] == "HASH"
    assert user["session_string"] == "SESS"
    assert user["gemini_api_key"] == "GKEY"


def test_auth_flow_secrets_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRETS_KEY", "unit-test-key")
    repo = _repo(tmp_path)
    repo.upsert_auth_flow(
        bot_user_id=3, chat_id=3, phone="+1", api_id=5, api_hash="AH",
        session_string="SS", phone_code_hash="PCH", status="awaiting_code",
    )
    with sqlite3.connect(tmp_path / "store.db") as conn:
        raw = conn.execute(
            "SELECT api_hash, session_string, phone_code_hash FROM auth_flows WHERE bot_user_id = 3"
        ).fetchone()
    assert all(v.startswith("enc::") for v in raw)

    flow = repo.get_auth_flow(bot_user_id=3)
    assert flow["api_hash"] == "AH"
    assert flow["session_string"] == "SS"
    assert flow["phone_code_hash"] == "PCH"


def test_disconnect_bot_user(tmp_path):
    repo = _repo(tmp_path)
    repo.upsert_bot_user(bot_user_id=7, chat_id=7, username="u", first_name="f")
    repo.save_bot_user_session(bot_user_id=7, phone="+100", api_id=1, api_hash="h", session_string="SESS", connected_at="now")
    assert repo.get_bot_user(bot_user_id=7)["session_string"] == "SESS"

    assert repo.disconnect_bot_user(bot_user_id=7) is True
    user = repo.get_bot_user(bot_user_id=7)
    assert user["session_string"] is None
    assert user["api_hash"] is None
    assert user["bot_user_id"] == 7  # row preserved
    # second disconnect is a no-op
    assert repo.disconnect_bot_user(bot_user_id=7) is False
