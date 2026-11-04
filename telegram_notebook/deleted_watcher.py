"""Always-on "recover deleted messages" watcher (the ``export-deleted`` module).

Telegram has **no API to fetch already-deleted messages**. The only way this can
work — for any tool, including the original Telegram-Toolset — is to keep a live
client that caches every incoming message and, when Telegram broadcasts a delete
update, surface the previously cached copy. Therefore it:

* only recovers messages cached **after** the watcher was enabled, and
* requires a persistent per-user Telethon client (opt-in via ``/watchdeleted on``).

The cache/recover *logic* is plain functions over the repository so it is
unit-testable; the Telethon event wiring + per-user thread lifecycle lives in
:class:`DeletedWatcherManager` and never crashes the bot on error.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from .toolset import _display_name as display_name

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def recover_deleted(
    repository: Any,
    *,
    owner_id: int,
    message_ids: list[int],
    chat_id: int | None,
    deleted_at: str | None = None,
) -> list[dict[str, Any]]:
    """Move any cached copies of ``message_ids`` into the recovered store.

    Returns the recovered rows (empty when none of the deleted ids were cached).
    """
    deleted_at = deleted_at or _now_iso()
    rows = repository.take_cached(owner_id=owner_id, message_ids=message_ids, chat_id=chat_id)
    if rows:
        repository.add_recovered(owner_id=owner_id, rows=rows, deleted_at=deleted_at)
    return rows


class DeletedWatcherManager:
    """Runs one persistent Telethon client per opted-in user, in its own thread.

    ``build_client`` is a callable ``(user_dict) -> TelegramClient`` so the
    Telethon dependency can be injected (and faked in tests). ``notify`` is an
    optional ``(owner_id, rows) -> None`` callback invoked when messages are
    recovered.
    """

    def __init__(
        self,
        *,
        repository: Any,
        build_client: Callable[[dict[str, Any]], Any],
        notify: Callable[[int, list[dict[str, Any]]], None] | None = None,
    ) -> None:
        self.repository = repository
        self.build_client = build_client
        self.notify = notify
        self._lock = threading.Lock()
        self._threads: dict[int, threading.Thread] = {}
        self._clients: dict[int, Any] = {}
        self._stopping: set[int] = set()

    def is_running(self, owner_id: int) -> bool:
        with self._lock:
            t = self._threads.get(owner_id)
            return bool(t and t.is_alive())

    def start_all(self) -> None:
        """Start watchers for every user who has the watcher enabled."""
        try:
            owners = self.repository.list_watch_deleted_owners()
        except Exception:
            logger.exception("Could not list deleted-watcher owners")
            return
        for owner in owners:
            self.start_for_user(owner)

    def start_for_user(self, user: dict[str, Any]) -> None:
        owner_id = int(user["bot_user_id"])
        with self._lock:
            if owner_id in self._threads and self._threads[owner_id].is_alive():
                return
            self._stopping.discard(owner_id)
            thread = threading.Thread(target=self._run, args=(user,), name=f"deleted-watcher-{owner_id}", daemon=True)
            self._threads[owner_id] = thread
        thread.start()
        logger.info("Started deleted-message watcher for user %s", owner_id)

    def stop_for_user(self, owner_id: int) -> None:
        with self._lock:
            self._stopping.add(owner_id)
            client = self._clients.pop(owner_id, None)
            self._threads.pop(owner_id, None)
        if client is not None:
            try:
                client.disconnect()
            except Exception:
                logger.debug("Error disconnecting watcher client for %s", owner_id, exc_info=True)
        logger.info("Stopped deleted-message watcher for user %s", owner_id)

    # -- internals ---------------------------------------------------------- #

    def _run(self, user: dict[str, Any]) -> None:
        import asyncio

        from telethon import events

        owner_id = int(user["bot_user_id"])
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            client = self.build_client(user)
        except Exception:
            logger.exception("Could not build watcher client for user %s", owner_id)
            return
        with self._lock:
            if owner_id in self._stopping:
                # stop_for_user ran before the client was registered — bail out now.
                self._threads.pop(owner_id, None)
                stopping = True
            else:
                self._clients[owner_id] = client
                stopping = False
        if stopping:
            try:
                client.disconnect()
            except Exception:
                logger.debug("Error disconnecting watcher client for %s", owner_id, exc_info=True)
            return

        @client.on(events.NewMessage())
        async def _on_new(event):  # pragma: no cover - needs a live client
            try:
                text = getattr(event.message, "message", "") or ""
                if not text:
                    return
                chat = await event.get_chat()
                sender = await event.get_sender()
                self.repository.cache_message(
                    owner_id=owner_id,
                    chat_id=event.chat_id,
                    message_id=event.id,
                    chat_title=display_name(chat),
                    sender=display_name(sender),
                    text=text,
                    message_date=str(getattr(event.message, "date", "")),
                )
            except Exception:
                logger.debug("watcher cache failed", exc_info=True)

        @client.on(events.MessageDeleted())
        async def _on_deleted(event):  # pragma: no cover - needs a live client
            try:
                rows = recover_deleted(
                    self.repository,
                    owner_id=owner_id,
                    message_ids=list(event.deleted_ids),
                    chat_id=event.chat_id,
                )
                if rows and self.notify:
                    self.notify(owner_id, rows)
            except Exception:
                logger.debug("watcher recover failed", exc_info=True)

        try:
            client.start()
            client.run_until_disconnected()
        except Exception:
            logger.exception("Deleted-message watcher for user %s stopped on error", owner_id)
        finally:
            with self._lock:
                self._clients.pop(owner_id, None)
                self._threads.pop(owner_id, None)
