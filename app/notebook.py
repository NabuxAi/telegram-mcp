"""Bridge to the NotebookLM-style archive engine (package ``telegram_notebook``).

The archive is keyed by the Telegram user id (``owner_id``), exactly as the
Telegram bot in that package keys it, so a person who connects here and a person
who talks to the bot share one archive. Heavy work (downloads, transcription,
embeddings) runs off the event loop: imports go through the package's JobWorker
thread with a dedicated Telethon connection built from the user's stored
session, and search/LLM calls are pushed to worker threads.
"""
import asyncio
import contextlib
import logging
import os
import re
from datetime import datetime, timezone
from typing import Annotated, Optional

from pydantic import Field

from . import config


def _bridge_env() -> None:
    """Map this server's config onto the environment names the notebook package reads."""
    pairs = {
        "TELEGRAM_API_ID": str(config.TG_API_ID) if config.TG_API_ID else "",
        "TELEGRAM_API_HASH": config.TG_API_HASH,
        "DATA_DIR": str(config.DATA_DIR),
        "DB_PATH": str(config.DB_PATH),
        "MEDIA_DIR": str(config.DATA_DIR / "media"),
        "SECRETS_KEY": config.APP_SECRET_KEY,
        "APP_NAME": config.APP_NAME,
    }
    for k, v in pairs.items():
        if v and not os.environ.get(k):
            os.environ[k] = v


_bridge_env()

from telegram_notebook import citations  # noqa: E402
from telegram_notebook.clustering import build_topics, label_cluster  # noqa: E402
from telegram_notebook.config import get_settings, model_for, provider_credentials  # noqa: E402
from telegram_notebook.db import Repository, connect  # noqa: E402
from telegram_notebook.embeddings import EmbeddingService  # noqa: E402
from telegram_notebook.export import build_markdown_export  # noqa: E402
from telegram_notebook.extraction import ExtractionService  # noqa: E402
from telegram_notebook.jobs import JobWorker  # noqa: E402
from telegram_notebook.llm import generate_text  # noqa: E402
from telegram_notebook.pipeline import FORWARDED_INBOX_URL, IngestionPipeline  # noqa: E402
from telegram_notebook.recent import recent_rows  # noqa: E402
from telegram_notebook.search import SearchService  # noqa: E402
from telegram_notebook.timeline import build_timeline  # noqa: E402
from telegram_notebook.transcription import TranscriptionService  # noqa: E402

from .db import Database, get_db  # noqa: E402
from .tg import ClientPool, chat_id_of, resolve  # noqa: E402

log = logging.getLogger("telegram_mcp.notebook")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Notebook:
    def __init__(self, db: Optional[Database] = None, pool: Optional[ClientPool] = None):
        self._db = db
        self.pool = pool
        self.settings = get_settings()
        self.repo = Repository(connect(self.settings.db_path))
        self.repo.init()
        self.worker = JobWorker(self.repo, self._run_job)
        self._started = False

    @property
    def db(self) -> Database:
        return self._db or get_db()

    # ---- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        if not self._started:
            self._started = True
            self.worker.start()

    def stop(self) -> None:
        if self._started:
            self.worker.stop()

    # ---- identity -----------------------------------------------------------
    def owner_id(self, user_id: int) -> int:
        u = self.db.get_user(user_id)
        if not u:
            raise ValueError("Account not connected.")
        return int(u["tg_user_id"])

    def bot_user(self, owner_id: int) -> Optional[dict]:
        return self.repo.get_bot_user(bot_user_id=owner_id)

    def sync_login(self, me, phone: str, session: str, api_id: Optional[int], api_hash: Optional[str]) -> None:
        """Mirror a web login into the notebook's bot_users table so bot commands work too."""
        try:
            self.repo.upsert_bot_user(bot_user_id=me.id, chat_id=me.id, username=me.username, first_name=me.first_name)
            custom = api_id and api_id != self.settings.telegram_api_id
            self.repo.save_bot_user_session(
                bot_user_id=me.id, phone=phone, api_id=api_id if custom else None,
                api_hash=api_hash if custom else None, session_string=session, connected_at=_now(),
            )
        except Exception:
            log.exception("sync_login failed for %s", getattr(me, "id", "?"))

    def sync_logout(self, tg_user_id: int) -> None:
        with contextlib.suppress(Exception):
            self.repo.disconnect_bot_user(bot_user_id=tg_user_id)

    def _session_for(self, owner_id: int) -> tuple[str, Optional[int], Optional[str]]:
        u = self.db.get_user_by_tg_id(owner_id)
        if u and u.get("session_ok"):
            s, api_id, api_hash = self.db.user_session(u["id"])
            return s, api_id, api_hash
        bu = self.bot_user(owner_id)
        if bu and bu.get("session_string"):
            return bu["session_string"], bu.get("api_id"), bu.get("api_hash")
        raise ValueError("No Telegram session for this account. Reconnect first.")

    # ---- per-user services ----------------------------------------------------
    def _key(self, user: Optional[dict], provider: str) -> Optional[str]:
        if provider == "nabugate":
            # Deliberately not per-user: the gateway token is the project's, and
            # spend is attributed to the project rather than to whoever asked.
            return self.settings.nabugate_api_key
        if provider == "gemini":
            return (user or {}).get("gemini_api_key") or self.settings.gemini_api_key
        if provider == "openai":
            return (user or {}).get("openai_api_key") or self.settings.openai_api_key
        return None

    def embeddings(self, user: Optional[dict]) -> EmbeddingService:
        s = self.settings
        # base_url follows the provider: the gateway URL for nabugate, Ollama's
        # for ollama, nothing for the cloud vendors. Passing Ollama's URL to the
        # OpenAI client for a gateway call is how requests went to localhost.
        _, base = provider_credentials(s, s.embedding_provider)
        return EmbeddingService(provider=s.embedding_provider, api_key=self._key(user, s.embedding_provider),
                                model=model_for(s, s.embedding_provider, "embedding"), base_url=base)

    def transcription(self, user: Optional[dict]) -> TranscriptionService:
        s = self.settings
        _, base = provider_credentials(s, s.transcription_provider)
        return TranscriptionService(provider=s.transcription_provider,
                                    api_key=self._key(user, s.transcription_provider),
                                    model=model_for(s, s.transcription_provider, "transcription"),
                                    base_url=base)

    def extraction(self, user: Optional[dict]) -> Optional[ExtractionService]:
        key = self._key(user, "gemini")
        if not key:
            return None
        return ExtractionService(provider="gemini", api_key=key, model="gemini-2.5-flash-lite")

    def search(self, user: Optional[dict]) -> SearchService:
        return SearchService(self.repo, self.embeddings(user))

    def llm_kwargs(self, user: Optional[dict]) -> dict:
        s = self.settings
        provider = (s.llm_provider or "ollama").lower()
        _, base = provider_credentials(s, provider)
        return {"provider": provider, "model": model_for(s, provider, "llm"), "base_url": base,
                "api_key": self._key(user, provider), "project_id": s.vertex_project_id,
                "region": s.vertex_region or "us-central1"}

    def llm_available(self, user: Optional[dict]) -> bool:
        provider = (self.settings.llm_provider or "ollama").lower()
        return provider in ("ollama", "local") or bool(self._key(user, provider))

    def pipeline(self, user: Optional[dict]) -> IngestionPipeline:
        return IngestionPipeline(settings=self.settings, repository=self.repo, transcription=self.transcription(user),
                                 embeddings=self.embeddings(user), extraction=self.extraction(user))

    # ---- import jobs (worker thread) ----------------------------------------------
    def _run_job(self, job: dict, on_progress, is_cancelled) -> None:
        owner_id = int(job["owner_id"])
        session, api_id, api_hash = self._session_for(owner_id)
        pipe = self.pipeline(self.bot_user(owner_id))

        async def _do():
            return await pipe.ingest_channel(
                owner_id=owner_id, channel_url=job["channel_url"], limit=job.get("limit_count"),
                api_id=api_id, api_hash=api_hash, session_string=session,
                resume_from=int(job.get("cursor") or 0), progress_cb=on_progress, should_cancel=is_cancelled,
            )
        asyncio.run(_do())

    def queue_import(self, owner_id: int, channel_url: str, limit: Optional[int]) -> int:
        return self.repo.create_job(owner_id=owner_id, channel_url=channel_url, limit=limit, created_at=_now())

    async def import_now(self, owner_id: int, channel_url: str, limit: Optional[int]) -> dict:
        session, api_id, api_hash = self._session_for(owner_id)
        pipe = self.pipeline(self.bot_user(owner_id))

        def _run():
            async def _do():
                return await pipe.ingest_channel(owner_id=owner_id, channel_url=channel_url, limit=limit,
                                                 api_id=api_id, api_hash=api_hash, session_string=session)
            return asyncio.run(_do())
        st = await asyncio.to_thread(_run)
        return {"source": st.channel_url, "title": st.channel_title, "messages_seen": st.processed_messages,
                "items_indexed": st.processed_media, "skipped_media": st.skipped_media}

    async def add_text(self, owner_id: int, *, source_label: str, text: str, key: int,
                       url: Optional[str], date: Optional[str]) -> bool:
        pipe = self.pipeline(self.bot_user(owner_id))

        def _run():
            return asyncio.run(pipe.ingest_forwarded_message(
                owner_id=owner_id, source_label=source_label, text=text, forward_key=key,
                message_url=url, message_date=date))
        return await asyncio.to_thread(_run)

    # ---- dashboard helper ---------------------------------------------------------
    def stats_text(self, owner_id: int) -> Optional[dict]:
        with contextlib.suppress(Exception):
            return self.repo.archive_stats(owner_id=owner_id)
        return None


# --------------------------------------------------------------------------------
# Source reference normalisation
# --------------------------------------------------------------------------------
def source_ref(value: Optional[str]) -> Optional[str]:
    """Map a chat reference to the canonical source URL used in the archive."""
    if not value:
        return None
    v = value.strip()
    if v.lower() in ("inbox", "notes", "forwarded", "saved"):
        return FORWARDED_INBOX_URL
    if v.startswith(("https://", "http://", "telegram://", "inbox://", "backup://")):
        return v.rstrip("/")
    if re.fullmatch(r"-?\d+", v):
        return f"telegram://channel/{v}"
    m = re.match(r"^(?:t\.me/)?@?([A-Za-z0-9_]{3,})/?$", v)
    if m:
        return f"https://t.me/{m.group(1)}"
    return v


def canonical_source(ent) -> str:
    username = getattr(ent, "username", None)
    if username:
        return f"https://t.me/{username}"
    return f"telegram://channel/{chat_id_of(ent)}"


def _ser_results(results) -> list[dict]:
    return [r.to_dict() for r in results]


# --------------------------------------------------------------------------------
# MCP tools
# --------------------------------------------------------------------------------
def register_tools(mcp, nb: Notebook, pool: ClientPool) -> None:
    from .tools import _err, _ok, content_hash, current_user_id, entry, tool

    Source = Annotated[Optional[str], Field(description="Restrict to one source: @username, chat id, t.me link, "
                                                        "the stored source URL, or 'inbox' for saved notes.")]
    Tag = Annotated[Optional[str], Field(description="Restrict to items carrying this tag.")]
    Account = Annotated[Optional[str], Field(
        description="Which of YOUR connected Telegram accounts this notebook belongs to: a 1-based "
                    "position from telegram_accounts_list, a @username, a phone, or an id. Omit for "
                    "the first connected account. Each account has its own separate archive.")]

    def owner() -> int:
        """The archive to act on: the Telegram id of the caller's selected account.

        current_user_id() reads the `account` the caller passed to this tool, so
        each connected account keeps its own separate notebook rather than every
        account of one person sharing a single archive.
        """
        return nb.owner_id(current_user_id())

    @tool("notebook_import_chat", title="Import chat into notebook", idempotent=True)
    async def notebook_import_chat(
        chat: Annotated[str, Field(description="@username, chat id, or t.me link of the channel/group/chat to index.")],
        limit: Annotated[Optional[int], Field(ge=1, description="Only the newest N messages; omit for the whole history.")] = None,
        wait: Annotated[bool, Field(description="True = run now and return stats (use for small limits, <= 200). "
                                                "False = queue a resumable background job and return its id.")] = False, account: Account = None) -> str:
        """Index a Telegram chat into the account's private notebook: every message's text plus
        media turned into text (audio/video transcribed, images/PDF OCR'd when a Gemini key is set,
        DOCX/XLSX parsed). Afterwards use notebook_search / notebook_ask on it. Re-importing is
        incremental (already-indexed items are skipped)."""
        en = await entry()
        ent = await resolve(en, chat)
        src = canonical_source(ent)
        oid = owner()
        if wait:
            if limit is None or limit > 200:
                return _err("For wait=True pass limit <= 200; queue larger imports with wait=False.")
            st = await nb.import_now(oid, src, limit)
            return _ok({"imported": st})
        job_id = nb.queue_import(oid, src, limit)
        return _ok({"job_id": job_id, "source": src, "status": "queued",
                    "hint": "Check progress with notebook_import_jobs; search works as soon as items land."})

    @tool("notebook_import_jobs", title="Import jobs", read_only=True, idempotent=True)
    async def notebook_import_jobs(limit: Annotated[int, Field(ge=1, le=50)] = 10, account: Account = None) -> str:
        """List background import jobs with status and progress."""
        jobs = await asyncio.to_thread(nb.repo.list_jobs, owner_id=owner(), limit=limit)
        return _ok({"jobs": jobs})

    @tool("notebook_cancel_import", title="Cancel import", idempotent=True)
    async def notebook_cancel_import(job_id: int, account: Account = None) -> str:
        """Ask a running/queued import job to stop."""
        ok = await asyncio.to_thread(nb.repo.request_job_cancel, owner_id=owner(), job_id=job_id)
        return _ok({"job_id": job_id, "cancel_requested": ok}) if ok else _err("No active job with that id.")

    @tool("notebook_index_messages", title="Index specific messages", idempotent=True)
    async def notebook_index_messages(
        chat: Annotated[str, Field(description="Chat the messages belong to.")],
        message_ids: Annotated[list[int], Field(min_length=1, max_length=100)], account: Account = None) -> str:
        """Add the text of specific messages (by id) from any chat to the notebook, without
        importing the whole chat. Good for 'remember this' or indexing search hits."""
        en = await entry()
        ent = await resolve(en, chat)
        oid = owner()
        msgs = await en.client.get_messages(ent, ids=message_ids)
        title = getattr(ent, "title", None) or getattr(ent, "first_name", None) or str(chat_id_of(ent))
        username = getattr(ent, "username", None)
        added = skipped = 0
        for m in msgs:
            if m is None or not (m.message or "").strip():
                skipped += 1
                continue
            url = f"https://t.me/{username}/{m.id}" if username else None
            key = content_hash("msg", str(chat_id_of(ent)), str(m.id))
            ok = await nb.add_text(oid, source_label=title, text=m.message, key=key, url=url,
                                   date=m.date.isoformat() if m.date else None)
            added += 1 if ok else 0
            skipped += 0 if ok else 1
        return _ok({"indexed": added, "skipped": skipped})

    @tool("notebook_save_note", title="Save note", idempotent=True)
    async def notebook_save_note(
        text: Annotated[str, Field(min_length=1, description="Text to remember.")],
        title: Annotated[Optional[str], Field(description="Short label for the note.")] = None,
        url: Annotated[Optional[str], Field(description="Optional link to attach as the note's source.")] = None, account: Account = None) -> str:
        """Store free text (a summary, a decision, a fact) in the notebook's inbox so later
        searches and questions can find it."""
        oid = owner()
        key = content_hash("note", text)
        ok = await nb.add_text(oid, source_label=title or "Note", text=text, key=key, url=url, date=_now())
        return _ok({"saved": ok, "note": None if ok else "An identical note already exists."})

    @tool("notebook_sources", title="Notebook sources", read_only=True, idempotent=True)
    async def notebook_sources(account: Account = None) -> str:
        """List the sources (chats/channels, saved notes inbox) indexed in the notebook."""
        rows = await asyncio.to_thread(nb.repo.list_channels, owner_id=owner())
        return _ok({"count": len(rows), "sources": rows})

    @tool("notebook_delete_source", title="Delete source", destructive=True, idempotent=True)
    async def notebook_delete_source(source: Annotated[str, Field(description="Source to remove (see notebook_sources).")], account: Account = None) -> str:
        """Remove a source and everything indexed from it. IRREVERSIBLE (re-import to rebuild)."""
        ok = await asyncio.to_thread(nb.repo.delete_channel_data, owner_id=owner(), channel_url=source_ref(source))
        return _ok({"deleted": ok}) if ok else _err("Source not found.")

    @tool("notebook_search", title="Search notebook", read_only=True, idempotent=True)
    async def notebook_search(
        query: Annotated[str, Field(min_length=1)],
        source: Source = None,
        tag: Tag = None,
        top_k: Annotated[int, Field(ge=1, le=30)] = 8, account: Account = None) -> str:
        """Semantic + keyword search over everything indexed in the notebook. Returns matching
        passages with their source and message link."""
        oid = owner()
        user = nb.bot_user(oid)
        res = await asyncio.to_thread(nb.search(user).search, owner_id=oid, query=query,
                                      channel_url=source_ref(source), tag=tag, top_k=top_k)
        return _ok({"count": len(res), "results": _ser_results(res)})

    @tool("notebook_ask", title="Ask notebook", read_only=True, idempotent=True)
    async def notebook_ask(
        question: Annotated[str, Field(min_length=1)],
        source: Source = None,
        tag: Tag = None,
        top_k: Annotated[int, Field(ge=1, le=20)] = 6, account: Account = None) -> str:
        """Answer a question from the notebook's indexed Telegram content using the server's LLM,
        grounded with numbered citations [n] that map to the returned sources. If you (the
        caller) are already a capable model, notebook_search + your own reasoning is cheaper."""
        oid = owner()
        user = nb.bot_user(oid)
        if not nb.llm_available(user):
            return _err("No LLM configured on the server (set LLM_PROVIDER / API key). Use notebook_search instead.")
        svc = nb.search(user)
        res = await asyncio.to_thread(svc.search, owner_id=oid, query=question, channel_url=source_ref(source),
                                      tag=tag, top_k=top_k)
        if not res:
            return _ok({"answer": None, "sources": [], "note": "Nothing relevant indexed yet. Import a chat first."})
        answer = await asyncio.to_thread(lambda: svc.generate_answer(query=question, results=res, **nb.llm_kwargs(user)))
        return _ok({"answer": answer, "cited": citations.cited_indices(answer, len(res)),
                    "sources": _ser_results(res)})

    @tool("notebook_summarize", title="Summarize", read_only=True, idempotent=True)
    async def notebook_summarize(source: Source = None, tag: Tag = None, account: Account = None) -> str:
        """LLM summary of the whole notebook, one source, or one tag."""
        oid = owner()
        user = nb.bot_user(oid)
        if not nb.llm_available(user):
            return _err("No LLM configured on the server.")
        items = await asyncio.to_thread(nb.repo.summary_items, owner_id=oid, channel_url=source_ref(source), tag=tag)
        label = tag or source or "the whole notebook"
        text = await asyncio.to_thread(lambda: nb.search(user).summarize(scope_label=label, items=items, **nb.llm_kwargs(user)))
        return _ok({"scope": label, "items": len(items), "summary": text})

    @tool("notebook_topics", title="Topics", read_only=True, idempotent=True)
    async def notebook_topics(source: Source = None, tag: Tag = None, account: Account = None) -> str:
        """Cluster indexed content into topics (offline, from stored embeddings)."""
        oid = owner()
        user = nb.bot_user(oid)
        items = await asyncio.to_thread(nb.repo.chunks_with_embeddings, owner_id=oid, channel_url=source_ref(source), tag=tag)
        if not items:
            return _ok({"topics": [], "note": "No embedded content yet."})
        namer = None
        if nb.llm_available(user):
            kw = nb.llm_kwargs(user)

            def namer(texts):  # noqa: F811
                return label_cluster(texts, generate=lambda p: generate_text(prompt=p, **kw))
        topics = await asyncio.to_thread(build_topics, items, namer=namer)
        return _ok({"topics": topics})

    @tool("notebook_timeline", title="Timeline", read_only=True, idempotent=True)
    async def notebook_timeline(
        source: Source = None, tag: Tag = None,
        granularity: Annotated[str, Field(description="'month' or 'day'.")] = "month", account: Account = None) -> str:
        """How many items were archived per month/day."""
        items = await asyncio.to_thread(nb.repo.timeline_items, owner_id=owner(), channel_url=source_ref(source), tag=tag)
        return _ok({"periods": build_timeline(items, granularity="day" if granularity == "day" else "month")})

    @tool("notebook_stats", title="Notebook stats", read_only=True, idempotent=True)
    async def notebook_stats(account: Account = None) -> str:
        """Overview of the notebook: items, sources, tags, media types, date range."""
        return _ok({"stats": await asyncio.to_thread(nb.repo.archive_stats, owner_id=owner())})

    @tool("notebook_recent", title="Recent items", read_only=True, idempotent=True)
    async def notebook_recent(limit: Annotated[int, Field(ge=1, le=100)] = 15, account: Account = None) -> str:
        """Most recently indexed items."""
        items = await asyncio.to_thread(nb.repo.timeline_items, owner_id=owner(), limit=limit)
        return _ok({"items": recent_rows(items, limit=limit)})

    @tool("notebook_get_item", title="Get item", read_only=True, idempotent=True)
    async def notebook_get_item(media_item_id: int, account: Account = None) -> str:
        """Full stored text of one indexed item (id from search results / recent)."""
        item = await asyncio.to_thread(nb.repo.get_media_item, owner_id=owner(), media_item_id=media_item_id)
        return _ok({"item": item}) if item else _err("Item not found.")

    @tool("notebook_export", title="Export markdown", read_only=True, idempotent=True)
    async def notebook_export(
        source: Source = None, tag: Tag = None,
        max_chars: Annotated[int, Field(ge=1000, le=400_000)] = 60_000, account: Account = None) -> str:
        """Export indexed content as Markdown (truncated to max_chars)."""
        items = await asyncio.to_thread(nb.repo.summary_items, owner_id=owner(), channel_url=source_ref(source), tag=tag, limit=2000)
        md = build_markdown_export(tag or source or "notebook", items, generated_at=_now())
        return _ok({"items": len(items), "markdown": md[:max_chars], "truncated": len(md) > max_chars})

    @tool("notebook_tags", title="Tags", read_only=True, idempotent=True)
    async def notebook_tags(account: Account = None) -> str:
        """List tags and how many items carry each."""
        return _ok({"tags": await asyncio.to_thread(nb.repo.list_tags, owner_id=owner())})

    @tool("notebook_rules", title="Tag rules", read_only=True, idempotent=True)
    async def notebook_rules(account: Account = None) -> str:
        """List auto-tagging rules (keyword -> tag)."""
        return _ok({"rules": await asyncio.to_thread(nb.repo.list_rules, owner_id=owner())})

    @tool("notebook_add_rule", title="Add tag rule", idempotent=True)
    async def notebook_add_rule(
        keyword: Annotated[str, Field(min_length=1, description="Case-insensitive substring to match.")],
        tag: Annotated[str, Field(min_length=1)], account: Account = None) -> str:
        """Add a rule: items whose text contains `keyword` get `tag` on import."""
        rid = await asyncio.to_thread(nb.repo.add_rule, owner_id=owner(), keyword=keyword, tag=tag, created_at=_now())
        return _ok({"rule_id": rid})

    @tool("notebook_remove_rule", title="Remove tag rule", destructive=True, idempotent=True)
    async def notebook_remove_rule(rule_id: int, account: Account = None) -> str:
        """Delete a tag rule by id."""
        ok = await asyncio.to_thread(nb.repo.remove_rule, owner_id=owner(), rule_id=rule_id)
        return _ok({"removed": ok}) if ok else _err("Rule not found.")
