"""One person, several Telegram accounts, addressed in the order they connected them."""

import os
import tempfile

import pytest

os.environ.setdefault("APP_SECRET_KEY", "x" * 48)
os.environ.setdefault("PUBLIC_BASE_URL", "https://tm.test")
os.environ.setdefault("TG_API_ID", "1")
os.environ.setdefault("TG_API_HASH", "h" * 32)

from app.db import Database  # noqa: E402


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "t.db"))


def _add(db, tg_id, owner=None, username=None, phone=None):
    return db.upsert_user(tg_user_id=tg_id, phone=phone or f"+9{tg_id}", first_name="A", last_name="B",
                          username=username, session_string="s", owner_key=owner)


def test_a_fresh_account_is_its_own_owner(db):
    uid = _add(db, 100)
    assert db.owner_of(uid) == "tg:100"
    assert [a["id"] for a in db.accounts_for_owner("tg:100")] == [uid]


def test_a_second_phone_joins_the_owner_in_connection_order(db):
    first = _add(db, 100)
    owner = db.owner_of(first)
    second = _add(db, 200, owner=owner)
    third = _add(db, 300, owner=owner)

    ids = [a["id"] for a in db.accounts_for_owner(owner)]
    assert ids == [first, second, third], "order must be the order of connection"
    assert [a["position"] for a in db.accounts_for_owner(owner)] == [0, 1, 2]


def test_relogging_an_existing_phone_never_moves_it_to_another_owner(db):
    """A browser session must not be able to claim a phone that is already someone's."""
    victim = _add(db, 100)                       # owner tg:100
    attacker = _add(db, 999)                     # owner tg:999
    # attacker's session re-logs the victim's phone, passing their own owner key
    again = db.upsert_user(tg_user_id=100, phone="+9100", first_name="A", last_name="B",
                           username=None, session_string="s2", owner_key=db.owner_of(attacker))
    assert again == victim
    assert db.owner_of(victim) == "tg:100", "existing account kept its owner"
    assert [a["id"] for a in db.accounts_for_owner("tg:999")] == [attacker]


def test_migration_is_idempotent_and_backfills_existing_rows(tmp_path):
    path = str(tmp_path / "old.db")
    d1 = Database(path)
    uid = _add(d1, 42)
    # simulate an older row with no owner
    d1._conn.execute("UPDATE users SET owner_key=NULL WHERE id=?", (uid,))
    d2 = Database(path)   # re-open runs the migration again
    assert d2.owner_of(uid) == "tg:42"
    d3 = Database(path)   # and again — must not fail on existing columns
    assert d3.owner_of(uid) == "tg:42"


def test_resolve_account_defaults_to_first_and_accepts_position_username_phone_id(db, monkeypatch):
    from app import tools

    first = _add(db, 100, username="alpha", phone="+98111")
    owner = db.owner_of(first)
    second = _add(db, 200, owner=owner, username="beta", phone="+98222")

    class _Pool:
        pass
    pool = _Pool(); pool.db = db
    monkeypatch.setattr(tools, "_pool", pool)
    monkeypatch.setattr(tools, "_token_user_id", lambda: first)

    assert tools.resolve_account(None) == first, "no account named -> first connected"
    assert tools.resolve_account("") == first
    assert tools.resolve_account("1") == first
    assert tools.resolve_account("2") == second
    assert tools.resolve_account("@beta") == second
    assert tools.resolve_account("beta") == second
    assert tools.resolve_account("+98222") == second
    assert tools.resolve_account(str(second)) == second
    assert tools.resolve_account("200") == second   # tg_user_id


def test_resolve_account_cannot_reach_another_owner_s_account(db, monkeypatch):
    from app import tools

    mine = _add(db, 100, username="mine")
    theirs = _add(db, 500, username="theirs")     # different owner

    class _Pool:
        pass
    pool = _Pool(); pool.db = db
    monkeypatch.setattr(tools, "_pool", pool)
    monkeypatch.setattr(tools, "_token_user_id", lambda: mine)

    with pytest.raises(ValueError):
        tools.resolve_account(str(theirs))
    with pytest.raises(ValueError):
        tools.resolve_account("@theirs")
    with pytest.raises(ValueError):
        tools.resolve_account("500")


def test_the_account_contextvar_is_scoped_to_one_tool_call(db, monkeypatch):
    """The wrapper sets it for the call and resets after — a leak would make the
    next call act on whatever account the previous caller named."""
    from app import tools

    first = _add(db, 100); owner = db.owner_of(first); second = _add(db, 200, owner=owner)

    class _Pool:
        pass
    pool = _Pool(); pool.db = db
    monkeypatch.setattr(tools, "_pool", pool)
    monkeypatch.setattr(tools, "_token_user_id", lambda: first)

    token = tools._account_ctx.set("2")
    try:
        assert tools.current_user_id() == second
    finally:
        tools._account_ctx.reset(token)
    assert tools.current_user_id() == first, "after reset, back to the default account"


def test_every_telegram_tool_accepts_an_account(monkeypatch):
    """A tool without `account` silently acts on the first account.

    That is the failure this whole change exists to prevent, and it comes back
    the moment someone adds a tool and forgets the parameter — so assert on the
    registered signatures rather than trusting review.
    """
    import inspect

    from app import tools

    registered = {}

    class _FakeMCP:
        def tool(self, name, annotations=None):
            def deco(fn):
                registered[name] = fn
                return fn
            return deco

    class _Pool:
        pass

    tools.register(_FakeMCP(), _Pool())
    assert registered, "no tools registered — the test would pass vacuously"

    missing = [
        n for n, fn in registered.items()
        if n != "telegram_accounts_list"
        and "account" not in inspect.signature(fn).parameters
    ]
    assert not missing, f"tools with no `account` parameter: {missing}"


def test_the_notebook_follows_the_selected_account(db, monkeypatch):
    """Each connected account keeps its own archive.

    The notebook keys everything by the Telegram user id it gets from
    current_user_id(), so if that ignored the caller's `account` the second
    account would read and write the first one's notebook.
    """
    from app import tools

    first = _add(db, 100)
    owner = db.owner_of(first)
    second = _add(db, 200, owner=owner)

    class _Pool:
        pass
    pool = _Pool(); pool.db = db
    monkeypatch.setattr(tools, "_pool", pool)
    monkeypatch.setattr(tools, "_token_user_id", lambda: first)

    class _NB:
        def __init__(self, d):
            self.db = d

        def owner_id(self, user_id):
            u = self.db.get_user(user_id)
            return int(u["tg_user_id"])

    nb = _NB(db)

    assert nb.owner_id(tools.current_user_id()) == 100
    token = tools._account_ctx.set("2")
    try:
        assert nb.owner_id(tools.current_user_id()) == 200, (
            "the notebook must follow the account the caller named"
        )
    finally:
        tools._account_ctx.reset(token)
