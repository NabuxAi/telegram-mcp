"""Telethon integration: per-user client pool, interactive login flows and the
helpers the MCP tools share (entity resolution, serialisation, safe URL fetch,
FloodWait handling).
"""
import asyncio
import ipaddress
import logging
import mimetypes
import re
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import unquote, urlsplit

import httpx
from telethon import TelegramClient, types, utils
from telethon.errors import (
    AuthKeyUnregisteredError,
    FloodWaitError,
    PhoneCodeExpiredError,
    PhoneCodeInvalidError,
    PhoneNumberInvalidError,
    PasswordHashInvalidError,
    SessionPasswordNeededError,
    SessionRevokedError,
    UserDeactivatedError,
    UserDeactivatedBanError,
)
from telethon.sessions import StringSession

from . import config
from .db import Database, get_db

log = logging.getLogger("telegram_mcp.tg")

SESSION_DEAD_ERRORS = (AuthKeyUnregisteredError, SessionRevokedError,
                       UserDeactivatedError, UserDeactivatedBanError)


class SessionInvalid(Exception):
    """The stored Telegram session no longer works; the user must reconnect."""


class LoginError(Exception):
    """User-facing login problem (wrong code, bad phone, ...)."""


# ----------------------------------------------------------------------------
# Client pool
# ----------------------------------------------------------------------------
@dataclass
class ClientEntry:
    user_id: int
    client: TelegramClient
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    entities: dict = field(default_factory=dict)
    last_used: float = field(default_factory=time.time)

    def touch(self):
        self.last_used = time.time()


def _new_client(session: str, api_id: int, api_hash: str) -> TelegramClient:
    return TelegramClient(
        StringSession(session or None), api_id, api_hash,
        device_model=config.APP_NAME, system_version="MCP", app_version="2.0",
        connection_retries=3, retry_delay=2, auto_reconnect=True,
    )


class ClientPool:
    def __init__(self, db: Optional[Database] = None):
        self._db = db
        self._entries: dict[int, ClientEntry] = {}
        self._locks: dict[int, asyncio.Lock] = {}

    @property
    def db(self) -> Database:
        return self._db or get_db()

    def _lock(self, user_id: int) -> asyncio.Lock:
        lock = self._locks.get(user_id)
        if lock is None:
            lock = self._locks[user_id] = asyncio.Lock()
        return lock

    async def get(self, user_id: int) -> ClientEntry:
        """Return a connected, authorised client for the user (creating it once)."""
        async with self._lock(user_id):
            entry = self._entries.get(user_id)
            if entry is None:
                user = self.db.get_user(user_id)
                if user is None:
                    raise SessionInvalid("This account is no longer connected. Reconnect it.")
                if not user["session_ok"]:
                    raise SessionInvalid("Telegram session expired or was revoked. Reconnect your account.")
                session, api_id, api_hash = self.db.user_session(user_id)
                entry = ClientEntry(user_id=user_id, client=_new_client(session, api_id, api_hash))
                self._entries[user_id] = entry
            c = entry.client
            try:
                if not c.is_connected():
                    await c.connect()
                if not await c.is_user_authorized():
                    raise SessionInvalid("Telegram session expired or was revoked. Reconnect your account.")
            except SESSION_DEAD_ERRORS as e:
                await self._kill(user_id, entry)
                raise SessionInvalid(f"Telegram session is no longer valid ({type(e).__name__}). Reconnect.") from e
            except SessionInvalid:
                await self._kill(user_id, entry)
                raise
            entry.touch()
            return entry

    async def adopt(self, user_id: int, client: TelegramClient) -> ClientEntry:
        """Take over an already-authorised client (from a login flow)."""
        async with self._lock(user_id):
            old = self._entries.pop(user_id, None)
            if old is not None and old.client is not client:
                try:
                    await old.client.disconnect()
                except Exception:
                    pass
            entry = ClientEntry(user_id=user_id, client=client)
            self._entries[user_id] = entry
            return entry

    async def _kill(self, user_id: int, entry: ClientEntry):
        self._entries.pop(user_id, None)
        self.db.mark_session(user_id, False)
        try:
            await entry.client.disconnect()
        except Exception:
            pass

    async def drop(self, user_id: int, logout: bool = False) -> None:
        async with self._lock(user_id):
            entry = self._entries.pop(user_id, None)
        if entry is None and logout:
            try:
                session, api_id, api_hash = self.db.user_session(user_id)
            except Exception:
                return
            entry = ClientEntry(user_id=user_id, client=_new_client(session, api_id, api_hash))
        if entry is None:
            return
        try:
            if logout:
                if not entry.client.is_connected():
                    await entry.client.connect()
                await entry.client.log_out()
            else:
                await entry.client.disconnect()
        except Exception as e:
            log.warning("drop(%s): %s", user_id, e)

    async def reaper(self, interval: int = 60):
        """Disconnect clients idle for longer than CLIENT_IDLE_SECONDS."""
        while True:
            await asyncio.sleep(interval)
            cutoff = time.time() - config.CLIENT_IDLE_SECONDS
            for uid, entry in list(self._entries.items()):
                if entry.last_used < cutoff and not entry.write_lock.locked():
                    async with self._lock(uid):
                        if self._entries.get(uid) is entry and entry.last_used < cutoff:
                            self._entries.pop(uid, None)
                            try:
                                await entry.client.disconnect()
                            except Exception:
                                pass

    async def close_all(self):
        for uid, entry in list(self._entries.items()):
            self._entries.pop(uid, None)
            try:
                await entry.client.disconnect()
            except Exception:
                pass

    def stats(self) -> dict:
        return {"connected_clients": len(self._entries)}


# ----------------------------------------------------------------------------
# Login flows (phone -> code -> optional 2FA password)
# ----------------------------------------------------------------------------
@dataclass
class LoginFlow:
    id: str
    client: TelegramClient
    phone: str
    api_id: int
    api_hash: str
    phone_code_hash: str = ""
    stage: str = "code"           # 'code' | 'password'
    txn: Optional[str] = None     # pending OAuth transaction id, if any
    # Who the finished account should belong to. Set when someone already
    # signed in connects another phone, so it joins their existing accounts
    # instead of becoming a second identity. None = the account is its own owner.
    owner_key: Optional[str] = None
    created: float = field(default_factory=time.time)
    attempts: int = 0


class LoginManager:
    def __init__(self, pool: ClientPool, db: Optional[Database] = None):
        self.pool = pool
        self._db = db
        self.flows: dict[str, LoginFlow] = {}
        self._ip_hits: dict[str, list[float]] = {}
        # Optional callback(me, phone, session_string, api_id, api_hash) after a login.
        self.on_login = None

    @property
    def db(self) -> Database:
        return self._db or get_db()

    def rate_limited(self, ip: str) -> bool:
        now = time.time()
        hits = [t for t in self._ip_hits.get(ip, []) if t > now - 3600]
        self._ip_hits[ip] = hits
        if len(hits) >= config.LOGIN_CODES_PER_HOUR:
            return True
        hits.append(now)
        return False

    def cleanup(self):
        cutoff = time.time() - config.LOGIN_FLOW_TTL
        for fid, flow in list(self.flows.items()):
            if flow.created < cutoff:
                self.flows.pop(fid, None)
                asyncio.ensure_future(_quiet_disconnect(flow.client))

    def get(self, flow_id: str) -> LoginFlow:
        self.cleanup()
        flow = self.flows.get(flow_id or "")
        if flow is None:
            raise LoginError("Login session expired. Start again.")
        return flow

    async def start(self, phone: str, txn: Optional[str] = None, owner_key: Optional[str] = None,
                    api_id: Optional[int] = None, api_hash: Optional[str] = None) -> LoginFlow:
        from .security import new_token
        phone = normalize_phone(phone)
        if config.ALLOWED_PHONES and phone not in config.ALLOWED_PHONES:
            raise LoginError("This phone number is not allowed on this server.")
        use_id = api_id or config.TG_API_ID
        use_hash = api_hash or config.TG_API_HASH
        client = _new_client("", use_id, use_hash)
        try:
            await client.connect()
            sent = await client.send_code_request(phone)
        except PhoneNumberInvalidError as e:
            await _quiet_disconnect(client)
            raise LoginError("Telegram rejected this phone number.") from e
        except FloodWaitError as e:
            await _quiet_disconnect(client)
            raise LoginError(f"Telegram asks to wait {e.seconds} seconds before requesting another code.") from e
        except Exception as e:
            await _quiet_disconnect(client)
            raise LoginError(f"Could not request a code: {type(e).__name__}: {e}") from e
        flow = LoginFlow(id=new_token(24), client=client, phone=phone, api_id=use_id, owner_key=owner_key,
                         api_hash=use_hash, phone_code_hash=sent.phone_code_hash, txn=txn)
        self.flows[flow.id] = flow
        return flow

    async def submit_code(self, flow_id: str, code: str) -> tuple[str, Optional[int]]:
        """Returns ('ok', user_id) or ('password', None) when 2FA is required."""
        flow = self.get(flow_id)
        code = re.sub(r"\D", "", code or "")
        if not code:
            raise LoginError("Enter the code Telegram sent you.")
        flow.attempts += 1
        if flow.attempts > 5:
            self.flows.pop(flow.id, None)
            await _quiet_disconnect(flow.client)
            raise LoginError("Too many attempts. Start again.")
        try:
            await flow.client.sign_in(phone=flow.phone, code=code, phone_code_hash=flow.phone_code_hash)
        except SessionPasswordNeededError:
            flow.stage = "password"
            return "password", None
        except PhoneCodeInvalidError as e:
            raise LoginError("Wrong code. Try again.") from e
        except PhoneCodeExpiredError as e:
            self.flows.pop(flow.id, None)
            await _quiet_disconnect(flow.client)
            raise LoginError("The code expired. Start again to get a new one.") from e
        except FloodWaitError as e:
            raise LoginError(f"Telegram asks to wait {e.seconds} seconds.") from e
        return "ok", await self._finish(flow)

    async def submit_password(self, flow_id: str, password: str) -> int:
        flow = self.get(flow_id)
        if flow.stage != "password":
            raise LoginError("A password was not requested for this login.")
        flow.attempts += 1
        if flow.attempts > 8:
            self.flows.pop(flow.id, None)
            await _quiet_disconnect(flow.client)
            raise LoginError("Too many attempts. Start again.")
        try:
            await flow.client.sign_in(password=password or "")
        except PasswordHashInvalidError as e:
            raise LoginError("Wrong two-step verification password.") from e
        except FloodWaitError as e:
            raise LoginError(f"Telegram asks to wait {e.seconds} seconds.") from e
        return await self._finish(flow)

    async def _finish(self, flow: LoginFlow) -> int:
        self.flows.pop(flow.id, None)
        me = await flow.client.get_me()
        session = flow.client.session.save()
        custom_creds = flow.api_id != config.TG_API_ID or flow.api_hash != config.TG_API_HASH
        user_id = self.db.upsert_user(
            tg_user_id=me.id, phone=flow.phone, first_name=me.first_name or "",
            last_name=me.last_name or "", username=me.username, session_string=session,
            api_id=flow.api_id if custom_creds else None,
            api_hash=flow.api_hash if custom_creds else None,
            # Someone already signed in who connects another phone keeps one
            # identity with several accounts; a fresh visitor becomes their own.
            owner_key=getattr(flow, "owner_key", None),
        )
        await self.pool.adopt(user_id, flow.client)
        if self.on_login is not None:
            try:
                self.on_login(me, flow.phone, session, flow.api_id, flow.api_hash)
            except Exception as e:  # never let a mirror failure break the login
                log.warning("on_login hook failed: %s", e)
        return user_id

    async def cancel(self, flow_id: str):
        flow = self.flows.pop(flow_id, None)
        if flow:
            await _quiet_disconnect(flow.client)


async def _quiet_disconnect(client: TelegramClient):
    try:
        await client.disconnect()
    except Exception:
        pass


def normalize_phone(phone: str) -> str:
    p = re.sub(r"[^\d+]", "", phone or "")
    if not p:
        raise LoginError("Enter your phone number with country code, e.g. +989121234567.")
    if p.startswith("00"):
        p = "+" + p[2:]
    if not p.startswith("+"):
        p = "+" + p
    if len(p) < 8:
        raise LoginError("Phone number looks too short.")
    return p


# ----------------------------------------------------------------------------
# Entity resolution
# ----------------------------------------------------------------------------
_ME = {"me", "self", "saved", "saved messages", "saved_messages"}


def normalize_target(target: str):
    """Turn the documented chat forms into something get_entity understands.

    Numeric ids must be ints: Telethon's string path runs parse_phone() first and
    would read "-1001234567890" as a phone number.
    """
    t = (target or "").strip()
    if t.lower() in _ME:
        return "me"
    if re.fullmatch(r"-?\d+", t):
        return int(t)
    if re.fullmatch(r"\+\d{7,16}", t):
        return t
    if t.startswith("@"):
        return t
    m = re.match(r"^(?:https?://)?(?:t\.me|telegram\.me|telegram\.dog)/(?:joinchat/|\+)?([\w\-]+)/?$", t)
    if m and not t.startswith("+"):
        return t if ("joinchat" in t or "/+" in t) else "@" + m.group(1)
    return t


async def resolve(entry: ClientEntry, target: Optional[str]):
    t = (target or "").strip()
    if not t:
        raise ValueError("`chat` is required (e.g. '@username', a numeric chat id, or 'me').")
    key = t.lower()
    if key in entry.entities:
        return entry.entities[key]
    norm = normalize_target(t)
    c = entry.client
    try:
        ent = await c.get_entity(norm)
    except ValueError:
        if isinstance(norm, int):
            # Not in the session cache yet: warm it up from the dialog list once.
            await c.get_dialogs(limit=None)
            ent = await c.get_entity(norm)
        else:
            raise
    entry.entities[key] = ent
    return ent


# ----------------------------------------------------------------------------
# FloodWait-aware write helper
# ----------------------------------------------------------------------------
async def guarded(entry: ClientEntry, coro_factory):
    """Run a mutating request, serialised per user, sleeping through short FloodWaits."""
    async with entry.write_lock:
        while True:
            try:
                return await coro_factory()
            except FloodWaitError as fw:
                if fw.seconds <= config.FLOODWAIT_CAP:
                    await asyncio.sleep(fw.seconds + 1)
                    continue
                raise RuntimeError(
                    f"FloodWait: Telegram asks to wait {fw.seconds}s (> cap {config.FLOODWAIT_CAP}s). "
                    "Pause and retry later."
                ) from fw


# ----------------------------------------------------------------------------
# Serialisation
# ----------------------------------------------------------------------------
def iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def display_name(ent) -> str:
    if ent is None:
        return ""
    if isinstance(ent, types.User):
        return " ".join(x for x in [ent.first_name, ent.last_name] if x) or (ent.username or str(ent.id))
    return getattr(ent, "title", None) or getattr(ent, "username", None) or str(getattr(ent, "id", ""))


def peer_kind(ent) -> str:
    if isinstance(ent, types.User):
        return "bot" if ent.bot else "user"
    if isinstance(ent, types.Chat):
        return "group"
    if isinstance(ent, types.Channel):
        return "supergroup" if ent.megagroup else "channel"
    return type(ent).__name__.lower()


def chat_id_of(ent) -> Optional[int]:
    try:
        return utils.get_peer_id(ent)
    except Exception:
        return getattr(ent, "id", None)


def ser_entity(ent, full=None) -> dict:
    d: dict[str, Any] = {
        "chat_id": chat_id_of(ent),
        "kind": peer_kind(ent),
        "title": display_name(ent),
        "username": getattr(ent, "username", None),
    }
    if isinstance(ent, types.User):
        d.update({
            "first_name": ent.first_name, "last_name": ent.last_name,
            "phone": ent.phone, "is_bot": ent.bot, "is_contact": ent.contact,
            "is_verified": ent.verified, "is_premium": bool(getattr(ent, "premium", False)),
            "is_deleted": ent.deleted,
        })
        if full is not None:
            d["about"] = getattr(full, "about", None)
            d["common_chats_count"] = getattr(full, "common_chats_count", None)
    else:
        d.update({
            "is_verified": bool(getattr(ent, "verified", False)),
            "is_creator": bool(getattr(ent, "creator", False)),
            "has_admin_rights": bool(getattr(ent, "admin_rights", None)),
            "is_restricted": bool(getattr(ent, "restricted", False)),
            "participants_count": getattr(ent, "participants_count", None),
        })
        if full is not None:
            d["about"] = getattr(full, "about", None)
            d["participants_count"] = getattr(full, "participants_count", d["participants_count"])
            d["pinned_msg_id"] = getattr(full, "pinned_msg_id", None)
            d["linked_chat_id"] = getattr(full, "linked_chat_id", None)
            d["can_view_participants"] = getattr(full, "can_view_participants", None)
    return d


def describe_media(m) -> Optional[dict]:
    if m.media is None:
        return None
    if m.photo:
        return {"type": "photo"}
    if m.document:
        doc = m.document
        info: dict[str, Any] = {"type": "document", "mime_type": doc.mime_type, "size": doc.size}
        for a in doc.attributes:
            if isinstance(a, types.DocumentAttributeFilename):
                info["file_name"] = a.file_name
            elif isinstance(a, types.DocumentAttributeVideo):
                info["type"] = "video_note" if a.round_message else "video"
                info["duration"] = a.duration
            elif isinstance(a, types.DocumentAttributeAudio):
                info["type"] = "voice" if a.voice else "audio"
                info["duration"] = a.duration
                if a.title:
                    info["title"] = a.title
                if a.performer:
                    info["performer"] = a.performer
            elif isinstance(a, types.DocumentAttributeSticker):
                info["type"] = "sticker"
                info["emoji"] = a.alt
            elif isinstance(a, types.DocumentAttributeAnimated):
                info["type"] = "gif"
        return info
    if m.web_preview:
        wp = m.web_preview
        return {"type": "webpage", "url": getattr(wp, "url", None), "title": getattr(wp, "title", None),
                "description": getattr(wp, "description", None)}
    if m.poll:
        p = m.poll.poll
        return {"type": "poll", "question": getattr(p.question, "text", p.question),
                "options": [getattr(a.text, "text", a.text) for a in p.answers],
                "closed": p.closed, "multiple_choice": p.multiple_choice, "quiz": p.quiz,
                "total_voters": getattr(m.poll.results, "total_voters", None)}
    if m.contact:
        return {"type": "contact", "phone": m.contact.phone_number,
                "first_name": m.contact.first_name, "last_name": m.contact.last_name}
    if m.geo:
        return {"type": "location", "lat": m.geo.lat, "long": m.geo.long}
    if m.dice:
        return {"type": "dice", "emoji": m.dice.emoticon, "value": m.dice.value}
    return {"type": type(m.media).__name__}


def ser_reactions(m) -> Optional[list]:
    r = getattr(m, "reactions", None)
    if not r or not r.results:
        return None
    out = []
    for rc in r.results:
        emoji = getattr(rc.reaction, "emoticon", None) or f"custom:{getattr(rc.reaction, 'document_id', '')}"
        out.append({"emoji": emoji, "count": rc.count, "chosen": bool(getattr(rc, "chosen_order", None) is not None)})
    return out


def ser_message(m, text_limit: Optional[int] = None) -> dict:
    text = m.message or ""
    if text_limit is not None and len(text) > text_limit:
        text = text[:text_limit] + "…"
    sender = getattr(m, "sender", None)
    d: dict[str, Any] = {
        "id": m.id,
        "date": iso(m.date),
        "chat_id": utils.get_peer_id(m.peer_id) if m.peer_id else None,
        "out": bool(m.out),
        "text": text,
        "text_len": len(m.message or ""),
        "sender": {"id": m.sender_id, "name": display_name(sender),
                   "username": getattr(sender, "username", None)} if m.sender_id else None,
        "reply_to_msg_id": m.reply_to_msg_id,
        "media": describe_media(m),
        "views": m.views,
        "forwards": m.forwards,
        "edit_date": iso(m.edit_date),
        "pinned": bool(m.pinned),
        "reactions": ser_reactions(m),
        "grouped_id": m.grouped_id,
        "mentioned": bool(m.mentioned),
    }
    if m.fwd_from:
        f = m.fwd_from
        d["forwarded_from"] = {"name": f.from_name or None,
                               "from_id": utils.get_peer_id(f.from_id) if f.from_id else None,
                               "date": iso(f.date), "channel_post": f.channel_post}
    if getattr(m, "via_bot_id", None):
        d["via_bot_id"] = m.via_bot_id
    if getattr(m, "action", None):
        d["service_action"] = type(m.action).__name__
    return {k: v for k, v in d.items() if v is not None and v is not False or k in ("text", "id")}


def ser_dialog(dlg, preview: int = 120) -> dict:
    ent = dlg.entity
    msg = dlg.message
    last = None
    if msg is not None:
        t = (msg.message or "")
        media = describe_media(msg)
        last = {"id": msg.id, "date": iso(msg.date), "out": bool(msg.out),
                "text": (t[:preview] + "…") if len(t) > preview else t,
                "media_type": media["type"] if media else None}
    return {
        "chat_id": dlg.id,
        "kind": peer_kind(ent),
        "title": dlg.name,
        "username": getattr(ent, "username", None),
        "unread_count": dlg.unread_count,
        "unread_mentions": dlg.unread_mentions_count,
        "pinned": bool(dlg.pinned),
        "archived": bool(dlg.archived),
        "last_message": last,
    }


# ----------------------------------------------------------------------------
# Safe remote fetch (SSRF-guarded, size/redirect limited)
# ----------------------------------------------------------------------------
def assert_public_host(host: str) -> None:
    """Refuse hosts that resolve to non-public addresses (server-side fetch)."""
    if config.ALLOW_PRIVATE_HOSTS:
        return
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ValueError(f"Cannot resolve host {host!r}: {e}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            raise ValueError(f"Host {host!r} resolves to non-public address {ip}; refusing to fetch.")


async def fetch_url(url: str, *, max_bytes: int, require_image: bool = False) -> tuple[bytes, str, str]:
    """Download `url`, returning (bytes, content_type, file_name)."""
    max_redirects = 3
    async with httpx.AsyncClient(follow_redirects=False, timeout=config.FILE_FETCH_TIMEOUT) as client:
        current = url
        for _ in range(max_redirects + 1):
            parts = urlsplit(current)
            if parts.scheme not in ("http", "https"):
                raise ValueError(f"URL must be http(s); got {parts.scheme or 'no'} scheme.")
            if not parts.hostname:
                raise ValueError("URL has no host.")
            assert_public_host(parts.hostname)
            async with client.stream("GET", current) as resp:
                if resp.is_redirect:
                    loc = resp.headers.get("location")
                    if not loc:
                        raise ValueError("Redirect without a Location header.")
                    current = str(resp.url.join(loc))
                    continue
                if resp.status_code != 200:
                    raise ValueError(f"URL returned HTTP {resp.status_code}.")
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if require_image and ctype and not ctype.startswith("image/"):
                    raise ValueError(f"URL returned {ctype}, expected image/*.")
                declared = resp.headers.get("content-length", "")
                if declared.isdigit() and int(declared) > max_bytes:
                    raise ValueError(f"File too large ({declared} bytes > {max_bytes}).")
                buf = bytearray()
                async for chunk in resp.aiter_bytes():
                    buf.extend(chunk)
                    if len(buf) > max_bytes:
                        raise ValueError(f"File too large (> {max_bytes} bytes); download aborted.")
                if not buf:
                    raise ValueError("URL returned an empty body.")
                name = _filename_from(resp.headers.get("content-disposition", ""), parts.path, ctype)
                return bytes(buf), ctype, name
        raise ValueError(f"Exceeded {max_redirects} redirects.")


def _filename_from(disposition: str, path: str, ctype: str) -> str:
    m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", disposition or "", re.I)
    if m:
        return unquote(m.group(1)).strip()
    base = unquote(path.rsplit("/", 1)[-1]) if path else ""
    if base and "." in base:
        return base
    ext = mimetypes.guess_extension(ctype or "") or ""
    return (base or "file") + ext


def parse_datetime(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as e:
        raise ValueError("Use ISO 8601 for dates, e.g. 2026-09-02T18:30:00+03:30") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_mode_of(v: Optional[str]):
    v = (v or "markdown").strip().lower()
    if v in ("none", "plain", "off", ""):
        return None
    if v in ("md", "markdown"):
        return "md"
    if v == "html":
        return "html"
    raise ValueError("parse_mode must be one of: markdown, html, none")
