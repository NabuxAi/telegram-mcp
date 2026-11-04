"""Telegram-Toolset features ported to our server-side (Telethon) architecture.

Each Telethon call is a thin ``async`` wrapper taking a connected ``client`` so
the bot/web layers stay thin; the pure formatters/builders below take plain
dicts and are unit-tested without any client.

Mirrors the modules of github.com/uburuntu/Telegram-Toolset:
``account-info``, ``scheduled``, ``resend`` and ``llm-export`` (the
``export-deleted`` module lives in :mod:`deleted_watcher`).
"""

from __future__ import annotations

import html
from typing import Any


def _display_name(entity: Any) -> str:
    """Best-effort human label for a Telethon user/chat entity."""
    if entity is None:
        return "Unknown"
    first = getattr(entity, "first_name", None)
    last = getattr(entity, "last_name", None)
    name = " ".join(p for p in (first, last) if p)
    return name or getattr(entity, "title", None) or getattr(entity, "username", None) or str(getattr(entity, "id", "Unknown"))


# --------------------------------------------------------------------------- #
# account-info
# --------------------------------------------------------------------------- #

async def account_info(client: Any) -> dict[str, Any]:
    """Return the connected account's profile + a couple of capability flags."""
    me = await client.get_me()
    return {
        "id": getattr(me, "id", None),
        "first_name": getattr(me, "first_name", None),
        "last_name": getattr(me, "last_name", None),
        "username": getattr(me, "username", None),
        "phone": getattr(me, "phone", None),
        "premium": bool(getattr(me, "premium", False)),
        "verified": bool(getattr(me, "verified", False)),
        "bot": bool(getattr(me, "bot", False)),
    }


def format_account(info: dict[str, Any]) -> str:
    """HTML summary of :func:`account_info` for a bot message."""
    name = " ".join(p for p in (info.get("first_name"), info.get("last_name")) if p) or "—"
    lines = [
        "<b>Account</b>",
        f"• Name: {html.escape(str(name))}",
        f"• Username: {('@' + info['username']) if info.get('username') else '—'}",
        f"• ID: {info.get('id') or '—'}",
        f"• Phone: {html.escape(str(info.get('phone'))) if info.get('phone') else '—'}",
        f"• Premium: {'✅' if info.get('premium') else '—'}",
        f"• Verified: {'✅' if info.get('verified') else '—'}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# scheduled
# --------------------------------------------------------------------------- #

async def list_scheduled(client: Any, peer: str, limit: int = 100) -> list[dict[str, Any]]:
    """Scheduled messages waiting to be sent in ``peer``."""
    messages = await client.get_messages(peer, scheduled=True, limit=limit)
    out = []
    for m in messages:
        out.append({
            "id": getattr(m, "id", None),
            "date": str(getattr(m, "date", "")) or None,
            "text": getattr(m, "message", "") or "",
        })
    return out


async def cancel_scheduled(client: Any, peer: str, message_ids: list[int]) -> int:
    """Delete scheduled messages by id. Returns how many were requested."""
    from telethon.tl.functions.messages import DeleteScheduledMessagesRequest

    entity = await client.get_input_entity(peer)
    await client(DeleteScheduledMessagesRequest(entity, message_ids))
    return len(message_ids)


def format_scheduled(peer_label: str, items: list[dict[str, Any]]) -> str:
    """HTML list of scheduled messages for a bot message."""
    if not items:
        return f"No scheduled messages in {html.escape(peer_label)}."
    lines = [f"<b>Scheduled — {html.escape(peer_label)}</b> ({len(items)})"]
    for it in items:
        when = it.get("date") or "?"
        text = (it.get("text") or "").strip().replace("\n", " ")
        snippet = (text[:80] + "…") if len(text) > 80 else text
        lines.append(f"• #{it.get('id')} · {html.escape(when)} — {html.escape(snippet) or '(media)'}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# resend
# --------------------------------------------------------------------------- #

async def resend_text(client: Any, target: str, text: str) -> int:
    """Send ``text`` to ``target`` on the user's behalf; return the new message id."""
    message = await client.send_message(target, text)
    return getattr(message, "id", 0)


# --------------------------------------------------------------------------- #
# llm-export
# --------------------------------------------------------------------------- #

async def fetch_chat_messages(client: Any, peer: str, limit: int = 200) -> tuple[str, list[dict[str, Any]]]:
    """Fetch up to ``limit`` recent messages from ``peer`` (newest-first from Telethon)."""
    entity = await client.get_entity(peer)
    chat_label = _display_name(entity)
    rows: list[dict[str, Any]] = []
    async for m in client.iter_messages(entity, limit=limit):
        text = getattr(m, "message", "") or ""
        if not text:
            continue
        sender = await m.get_sender() if hasattr(m, "get_sender") else None
        rows.append({
            "id": getattr(m, "id", None),
            "date": str(getattr(m, "date", "")) or None,
            "sender": _display_name(sender),
            "text": text,
        })
    return chat_label, rows


def build_llm_export(chat_label: str, rows: list[dict[str, Any]]) -> str:
    """Render fetched messages into an assistant-friendly Markdown transcript.

    ``rows`` come newest-first (Telethon order); the transcript is emitted
    oldest-first so it reads like a conversation.
    """
    lines = [f"# Telegram chat — {chat_label}", "", f"_{len(rows)} message(s)_", ""]
    for r in reversed(rows):
        when = r.get("date") or ""
        sender = r.get("sender") or "Unknown"
        head = f"**{sender}**" + (f" · {when}" if when else "")
        lines.append(head)
        lines.append((r.get("text") or "").strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
