from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .config import get_settings, model_for, provider_credentials, upsert_env_values
from .db import Repository, connect
from .embeddings import EmbeddingService
from .extraction import ExtractionService
from .logging_config import setup_logging
from .model_catalog import ModelCatalogService
from .pipeline import IngestionPipeline, build_media_text_extractor
from .recent import recent_rows
from .search import SearchService
from .telegram_backup import (
    count_messages,
    make_zip_file_resolver,
    parse_export,
    read_export,
    render_markdown,
)
from .timeline import build_timeline
from .transcription import TranscriptionService
from .webapp_auth import init_data_user_id, verify_init_data

logger = logging.getLogger(__name__)

# The web dashboard has no per-user login, so all of its data lives under a
# single fixed owner id. This keeps the dashboard's archive isolated from the
# per-user (bot_user_id) archives created through the Telegram bot.
WEB_OWNER_ID = 0


@dataclass(slots=True)
class RuntimeConfig:
    transcription_provider: str
    transcription_model: str
    embedding_provider: str
    embedding_model: str
    llm_provider: str
    llm_model: str
    ollama_base_url: str
    openai_enabled: bool
    gemini_enabled: bool


class AppState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.catalog = ModelCatalogService()
        self.reload()

    def reload(self) -> None:
        with self.lock:
            get_settings.cache_clear()
            self.settings = get_settings()
            self.repository = Repository(connect(self.settings.db_path))
            self.repository.init()
            embed_key, embed_base = provider_credentials(self.settings, self.settings.embedding_provider)
            self.embeddings = EmbeddingService(
                provider=self.settings.embedding_provider,
                api_key=embed_key,
                model=model_for(self.settings, self.settings.embedding_provider, "embedding"),
                base_url=embed_base,
            )
            tr_key2, tr_base2 = provider_credentials(self.settings, self.settings.transcription_provider)
            self.transcription = TranscriptionService(
                provider=self.settings.transcription_provider,
                api_key=self._api_key_for(self.settings.transcription_provider),
                model=model_for(self.settings, self.settings.transcription_provider, "transcription"),
                base_url=tr_base2,
            )
            # OCR/PDF extraction is Gemini-only; DOCX/XLSX are parsed locally.
            self.extraction = ExtractionService(
                provider="gemini",
                api_key=self.settings.gemini_api_key,
                model=self.settings.transcription_model,
            )
            self.pipeline = IngestionPipeline(
                settings=self.settings,
                repository=self.repository,
                transcription=self.transcription,
                embeddings=self.embeddings,
                extraction=self.extraction,
            )
            self.search_service = SearchService(self.repository, self.embeddings)

    def _api_key_for(self, provider: str) -> str | None:
        if provider == "gemini":
            return self.settings.gemini_api_key
        if provider == "openai":
            return self.settings.openai_api_key
        return None

    def vertex_search_config(self) -> dict[str, object] | None:
        """Vertex AI Search config from settings, or None to use local search."""
        s = self.settings
        if s.vertex_project_id and s.vertex_region and s.vertex_endpoint_id and s.vertex_deployed_index_id:
            return {
                "api_key": s.gemini_api_key,
                "project_id": s.vertex_project_id,
                "region": s.vertex_region,
                "index_endpoint_id": s.vertex_endpoint_id,
                "deployed_index_id": s.vertex_deployed_index_id,
            }
        return None

    def runtime_config(self) -> RuntimeConfig:
        return RuntimeConfig(
            transcription_provider=self.settings.transcription_provider,
            transcription_model=self.settings.transcription_model,
            embedding_provider=self.settings.embedding_provider,
            embedding_model=self.settings.embedding_model,
            llm_provider=self.settings.llm_provider,
            llm_model=self.settings.llm_model,
            ollama_base_url=self.settings.ollama_base_url,
            openai_enabled=bool(self.settings.openai_api_key),
            gemini_enabled=bool(self.settings.gemini_api_key),
        )

    def llm_params(self) -> dict[str, object]:
        """Generation kwargs for the configured chat/RAG provider (default: Ollama)."""
        s = self.settings
        return {
            "provider": s.llm_provider,
            "model": s.llm_model,
            "base_url": s.ollama_base_url,
            "api_key": self._api_key_for(s.llm_provider),
            "project_id": s.vertex_project_id,
            "region": s.vertex_region or "us-central1",
        }

    def list_models(self, *, provider: str, capability: str | None) -> list[dict[str, object]]:
        provider = provider.lower()
        api_key = self._api_key_for(provider)
        mapped_capability = capability
        if provider == "gemini":
            if capability == "transcription":
                mapped_capability = "generateContent"
            elif capability == "embedding":
                mapped_capability = "embedContent"
        return self.catalog.list_models(
            provider=provider,
            api_key=api_key,
            capability=mapped_capability,
        )


state = AppState()


def _run_account_op(coro_factory):
    """Run a Telethon coroutine against the server's connected account.

    The web dashboard has a single connected account (``TELEGRAM_SESSION_STRING``),
    so the Telegram-Toolset endpoints operate on it. Builds an on-demand client,
    runs the coroutine, and disconnects.
    """
    settings = state.settings
    if not settings.telegram_session_string:
        raise RuntimeError("Server account not connected. Set TELEGRAM_SESSION_STRING.")
    from .telegram_client import build_client_from_session_string

    client = build_client_from_session_string(
        settings, settings.telegram_session_string,
        api_id=settings.telegram_api_id, api_hash=settings.telegram_api_hash,
    )

    async def _do():
        async with client:
            return await coro_factory(client)

    return asyncio.run(_do())


INDEX_HTML = """
<!doctype html>
<html lang="fa" dir="rtl">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Telegram Notebook — پنل کنترل سینمایی</title>
    <meta name="theme-color" content="#06070b" />
    <meta name="description" content="پنل هوشمند تبدیل آرشیو تلگرام به یک حافظه‌ی قابل‌جستجو با جست‌وجوی معنایی و پرسش‌وپاسخ RAG." />
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500&family=Inter:wght@400;500;600&family=Vazirmatn:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
    <style>
      :root {
        --bg-0: #06070b;
        --bg-1: #0a0d14;
        --ink: #f3efe6;
        --muted: #98a1b2;
        --gold: #e9cd97;
        --gold-soft: #f3ddae;
        --emerald: #34d8a8;
        --teal: #14b894;
        --line: rgba(233, 205, 151, 0.16);
        --line-soft: rgba(255, 255, 255, 0.07);
        --card: rgba(15, 20, 30, 0.62);
        --radius: 22px;
        --shadow: 0 30px 80px rgba(0, 0, 0, 0.55);
      }
      * { box-sizing: border-box; }
      html { scroll-behavior: smooth; }
      body {
        margin: 0;
        min-height: 100vh;
        font-family: "Vazirmatn", "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
        color: var(--ink);
        background: var(--bg-0);
        overflow-x: hidden;
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
      }
      ::selection { background: rgba(233, 205, 151, 0.28); color: #fff; }

      /* ---------- cinematic background ---------- */
      .bg { position: fixed; inset: 0; z-index: -1; overflow: hidden; }
      .bg-aurora {
        position: absolute; inset: -25%;
        background:
          radial-gradient(40% 40% at 18% 12%, rgba(20, 184, 148, 0.22), transparent 60%),
          radial-gradient(38% 38% at 82% 16%, rgba(233, 205, 151, 0.20), transparent 60%),
          radial-gradient(55% 55% at 50% 108%, rgba(36, 80, 120, 0.30), transparent 65%),
          linear-gradient(180deg, #06070b 0%, #080b12 45%, #06070b 100%);
        filter: saturate(1.15);
        animation: drift 28s ease-in-out infinite alternate;
      }
      @keyframes drift {
        0% { transform: translate3d(0, 0, 0) scale(1); }
        100% { transform: translate3d(0, -3%, 0) scale(1.07); }
      }
      .bg-grid {
        position: absolute; inset: 0; opacity: 0.5;
        background-image:
          linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px);
        background-size: 64px 64px;
        -webkit-mask-image: radial-gradient(circle at 50% 28%, #000 0%, transparent 72%);
        mask-image: radial-gradient(circle at 50% 28%, #000 0%, transparent 72%);
      }
      #fx { position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0.6; }
      .bg-grain {
        position: absolute; inset: 0; pointer-events: none; opacity: 0.05; mix-blend-mode: overlay;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
      }
      .bg-vignette {
        position: absolute; inset: 0; pointer-events: none;
        background: radial-gradient(120% 90% at 50% 0%, transparent 55%, rgba(0, 0, 0, 0.6) 100%);
      }

      /* ---------- layout ---------- */
      .wrap { max-width: 1200px; margin: 0 auto; padding: 26px 20px 96px; }

      /* ---------- topbar ---------- */
      .topbar {
        position: sticky; top: 0; z-index: 30;
        backdrop-filter: blur(16px);
        background: linear-gradient(180deg, rgba(6, 7, 11, 0.86), rgba(6, 7, 11, 0.3));
        border-bottom: 1px solid var(--line-soft);
      }
      .topbar-inner {
        max-width: 1200px; margin: 0 auto; padding: 14px 20px;
        display: flex; align-items: center; justify-content: space-between; gap: 14px;
      }
      .brand { display: flex; align-items: center; gap: 11px; font-weight: 600; }
      .brand-mark {
        display: grid; place-items: center; width: 36px; height: 36px; border-radius: 12px;
        font-size: 1.1rem; color: var(--gold);
        background: linear-gradient(135deg, rgba(52, 216, 168, 0.25), rgba(233, 205, 151, 0.25));
        border: 1px solid var(--line); box-shadow: 0 0 28px rgba(52, 216, 168, 0.25);
      }
      .brand-name { font-size: 1.06rem; letter-spacing: 0.2px; }
      .brand-name em { font-style: normal; color: var(--gold); }
      .topnav { display: flex; align-items: center; gap: 22px; }
      .topnav a { color: var(--muted); text-decoration: none; font-size: 0.92rem; transition: color 0.2s; }
      .topnav a:hover { color: var(--ink); }
      .topnav .pill {
        padding: 8px 16px; border-radius: 999px; color: var(--ink);
        border: 1px solid var(--line); background: rgba(255, 255, 255, 0.03);
      }
      .topnav .pill:hover { border-color: var(--gold); color: var(--gold); }

      /* ---------- hero ---------- */
      .hero { padding: 66px 6px 30px; text-align: center; }
      .eyebrow {
        display: inline-block; padding: 7px 17px; border-radius: 999px; margin-bottom: 22px;
        font-size: 0.8rem; letter-spacing: 0.6px; color: var(--gold-soft);
        border: 1px solid var(--line); background: rgba(233, 205, 151, 0.06);
      }
      .hero h1 {
        margin: 0; font-weight: 600;
        font-size: clamp(2.1rem, 5.2vw, 4.1rem); line-height: 1.16; letter-spacing: -0.01em;
      }
      .hero .grad {
        background: linear-gradient(100deg, var(--emerald), var(--gold) 58%, var(--gold-soft));
        -webkit-background-clip: text; background-clip: text; color: transparent;
      }
      .hero p {
        max-width: 60ch; margin: 22px auto 0; color: var(--muted);
        font-size: 1.05rem; line-height: 2;
      }
      .hero-stats { display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; margin-top: 32px; }
      .chip {
        display: flex; align-items: baseline; gap: 9px; padding: 12px 18px; border-radius: 15px;
        border: 1px solid var(--line-soft); background: var(--card); backdrop-filter: blur(8px);
      }
      .chip b { font-size: 1.12rem; color: var(--gold); font-weight: 600; }
      .chip span { font-size: 0.82rem; color: var(--muted); }

      /* ---------- grid + cards ---------- */
      .grid {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
        gap: 18px; margin-top: 48px;
      }
      .card {
        position: relative; overflow: hidden;
        background: var(--card); border: 1px solid var(--line-soft);
        border-radius: var(--radius); padding: 24px;
        box-shadow: var(--shadow); backdrop-filter: blur(18px);
        transition: transform 0.4s cubic-bezier(.2, .7, .2, 1), border-color 0.4s, box-shadow 0.4s;
      }
      .card::before {
        content: ""; position: absolute; inset: 0; border-radius: inherit; padding: 1px;
        background: linear-gradient(140deg, rgba(233, 205, 151, 0.45), transparent 42%, rgba(52, 216, 168, 0.3));
        -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
        -webkit-mask-composite: xor; mask-composite: exclude;
        opacity: 0; transition: opacity 0.4s; pointer-events: none;
      }
      .card:hover { transform: translateY(-4px); border-color: transparent; box-shadow: 0 42px 90px rgba(0, 0, 0, 0.62); }
      .card:hover::before { opacity: 1; }
      .card.span { grid-column: 1 / -1; }
      .card-head { display: flex; align-items: center; gap: 14px; margin-bottom: 18px; scroll-margin-top: 90px; }
      .card-icon {
        flex: 0 0 auto; width: 46px; height: 46px; border-radius: 14px; display: grid; place-items: center;
        font-size: 1.3rem; color: var(--gold);
        background: linear-gradient(135deg, rgba(52, 216, 168, 0.16), rgba(233, 205, 151, 0.14));
        border: 1px solid var(--line);
      }
      .card-head h2 { margin: 0; font-size: 1.16rem; font-weight: 600; letter-spacing: 0.2px; }
      .card-head .sub { margin: 3px 0 0; font-size: 0.8rem; color: var(--muted); }

      /* ---------- forms ---------- */
      .settings-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }
      label { display: block; margin: 14px 0 7px; font-size: 0.82rem; color: var(--muted); }
      .card-head + label { margin-top: 2px; }
      .settings-grid label { margin-top: 0; }
      input, textarea, select {
        width: 100%; border-radius: 14px; padding: 12px 14px; font: inherit;
        color: var(--ink); background: rgba(255, 255, 255, 0.04); border: 1px solid var(--line-soft);
        transition: border-color 0.2s, box-shadow 0.2s, background 0.2s;
      }
      input::placeholder, textarea::placeholder { color: rgba(152, 161, 178, 0.6); }
      input:focus, textarea:focus, select:focus {
        outline: none; border-color: var(--gold);
        box-shadow: 0 0 0 3px rgba(233, 205, 151, 0.13); background: rgba(255, 255, 255, 0.06);
      }
      select option { background: #0b0f16; color: var(--ink); }
      textarea { min-height: 120px; resize: vertical; line-height: 1.8; }
      input[type="file"] { padding: 10px; color: var(--muted); }
      input[type="file"]::file-selector-button {
        margin-inline-end: 12px; padding: 9px 14px; border-radius: 10px; border: 1px solid var(--line);
        background: rgba(233, 205, 151, 0.08); color: var(--gold); font: inherit; cursor: pointer;
      }
      .btn-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 16px; }
      .btn-row button { width: auto; }
      button {
        width: 100%; border: none; cursor: pointer; font: inherit; font-weight: 600;
        padding: 12px 22px; border-radius: 14px; color: #06120e; letter-spacing: 0.2px;
        background: linear-gradient(120deg, var(--emerald), var(--teal));
        box-shadow: 0 10px 30px rgba(20, 184, 148, 0.28);
        transition: transform 0.2s, box-shadow 0.2s, filter 0.2s;
      }
      button:hover { transform: translateY(-2px); box-shadow: 0 16px 40px rgba(20, 184, 148, 0.42); filter: brightness(1.06); }
      button:active { transform: translateY(0); }
      button.secondary {
        color: #1a1206; background: linear-gradient(120deg, var(--gold-soft), var(--gold));
        box-shadow: 0 10px 30px rgba(233, 205, 151, 0.25);
      }
      button.secondary:hover { box-shadow: 0 16px 40px rgba(233, 205, 151, 0.42); }

      /* ---------- results / status ---------- */
      .results { margin-top: 18px; display: grid; gap: 12px; }
      .result {
        padding: 16px; border-radius: 16px; line-height: 1.9;
        background: rgba(255, 255, 255, 0.03); border: 1px solid var(--line-soft);
      }
      .meta { font-size: 0.82rem; color: var(--muted); margin-bottom: 8px; }
      .meta strong { color: var(--gold); }
      .status { min-height: 24px; margin-top: 12px; font-size: 0.9rem; color: var(--emerald); }
      .tiny { font-size: 0.84rem; color: var(--muted); line-height: 1.9; }
      a { color: var(--gold); }
      code { background: rgba(255, 255, 255, 0.06); padding: 1px 6px; border-radius: 6px; font-size: 0.85em; }
      #brainAnswer h3 { margin: 0 0 10px; font-size: 1.1rem; }
      #answerText { white-space: pre-wrap; color: var(--ink); line-height: 2; }

      /* ---------- reveal animation ---------- */
      .reveal { opacity: 0; transform: translateY(22px); transition: opacity 0.7s ease, transform 0.7s cubic-bezier(.2, .7, .2, 1); }
      .reveal.in { opacity: 1; transform: none; }

      ::-webkit-scrollbar { width: 11px; }
      ::-webkit-scrollbar-track { background: #06070b; }
      ::-webkit-scrollbar-thumb { background: linear-gradient(var(--teal), var(--gold)); border-radius: 99px; border: 3px solid #06070b; }

      @media (max-width: 640px) {
        .topnav { gap: 13px; }
        .topnav a:not(.pill) { display: none; }
        .hero { padding-top: 42px; }
      }
    </style>
  </head>
  <body>
    <div class="bg" aria-hidden="true">
      <div class="bg-aurora"></div>
      <div class="bg-grid"></div>
      <canvas id="fx"></canvas>
      <div class="bg-grain"></div>
      <div class="bg-vignette"></div>
    </div>

    <header class="topbar">
      <div class="topbar-inner">
        <div class="brand">
          <span class="brand-mark">◎</span>
          <span class="brand-name">Telegram <em>Notebook</em></span>
        </div>
        <nav class="topnav">
          <a href="/">خانه</a>
          <a href="#settings">تنظیمات</a>
          <a href="#library">کتابخانه</a>
          <a class="pill" href="https://github.com/shm379/telegram-notebooklm-mvp" target="_blank" rel="noreferrer">GitHub</a>
        </nav>
      </div>
    </header>

    <main class="wrap">
      <section class="hero reveal">
        <span class="eyebrow">پنل کنترل · نوت‌بوک هوشمند تلگرام</span>
        <h1>آرشیو تلگرام شما،<br /><span class="grad">یک حافظه‌ی سینمایی و قابل‌جستجو.</span></h1>
        <p>
          کانال‌ها، چت‌ها و فایل‌های بکاپ را وارد کن؛ ویدیو و صوت به متن تبدیل می‌شوند و
          مثل یک NotebookLM شخصی روی آرشیوت جست‌وجوی معنایی و پرسش‌وپاسخ مستند داری —
          با امکان جابه‌جایی آزاد بین OpenAI و Gemini.
        </p>
        <div class="hero-stats">
          <div class="chip"><b>۲</b><span>موتور OpenAI / Gemini</span></div>
          <div class="chip"><b>RAG</b><span>پاسخ مستند و ارجاع‌دار</span></div>
          <div class="chip"><b>∞</b><span>آرشیو قابل‌جستجو</span></div>
        </div>
      </section>

      <section class="grid">
        <div class="card span reveal">
          <div class="card-head" id="settings">
            <div class="card-icon">⚙️</div>
            <div>
              <h2>تنظیمات موتور</h2>
              <p class="sub">Provider، مدل و کلیدهای API</p>
            </div>
          </div>
          <div class="settings-grid">
            <div>
              <label for="transcriptionProvider">Transcription Provider</label>
              <select id="transcriptionProvider">
                <option value="openai">OpenAI</option>
                <option value="gemini">Gemini</option>
              </select>
            </div>
            <div>
              <label for="transcriptionModel">Transcription Model</label>
              <select id="transcriptionModel"></select>
            </div>
            <div>
              <label for="embeddingProvider">Embedding Provider</label>
              <select id="embeddingProvider">
                <option value="openai">OpenAI</option>
                <option value="gemini">Gemini</option>
              </select>
            </div>
            <div>
              <label for="embeddingModel">Embedding Model</label>
              <select id="embeddingModel"></select>
            </div>
            <div>
              <label for="geminiApiKey">Gemini API Key</label>
              <input id="geminiApiKey" type="password" placeholder="اختیاری؛ فقط برای بروزرسانی" />
            </div>
            <div>
              <label for="openaiApiKey">OpenAI API Key</label>
              <input id="openaiApiKey" type="password" placeholder="اختیاری؛ فقط برای بروزرسانی" />
            </div>
          </div>
          <div class="btn-row">
            <button id="reloadModelsBtn" type="button">بارگذاری مدل‌ها</button>
            <button id="saveSettingsBtn" class="secondary" type="button">ذخیره تنظیمات</button>
          </div>
          <div class="tiny" id="settingsSummary"></div>
          <div class="status" id="settingsStatus"></div>
        </div>

        <div class="card reveal">
          <div class="card-head">
            <div class="card-icon">📡</div>
            <div>
              <h2>دریافت کانال</h2>
              <p class="sub">ingest پیام‌ها از یک کانال عمومی</p>
            </div>
          </div>
          <label for="channelUrl">Channel URL</label>
          <input id="channelUrl" value="https://t.me/example_channel" />
          <label for="limit">Recent posts limit</label>
          <input id="limit" type="number" value="50" min="1" max="500" />
          <button id="ingestBtn">شروع دریافت</button>
          <div class="status" id="ingestStatus"></div>
        </div>

        <div class="card reveal">
          <div class="card-head">
            <div class="card-icon">🔎</div>
            <div>
              <h2>جست‌وجو و پرسش</h2>
              <p class="sub">جست‌وجوی معنایی + پاسخ RAG مستند</p>
            </div>
          </div>
          <label for="query">Query / Question</label>
          <textarea id="query">هوش مصنوعی و مدل‌های زبانی</textarea>
          <label for="searchChannel">Optional channel filter</label>
          <input id="searchChannel" placeholder="https://t.me/example_channel" />
          <div class="btn-row">
            <button class="secondary" id="searchBtn">جست‌وجوی متن</button>
            <button id="askBtn">پرسش از مغز AI</button>
          </div>
          <div class="status" id="searchStatus"></div>
        </div>

        <div class="card reveal">
          <div class="card-head" id="library">
            <div class="card-icon">🗂️</div>
            <div>
              <h2>کتابخانه</h2>
              <p class="sub">نمای کلی، timeline ماهانه و آخرین موارد</p>
            </div>
          </div>
          <p class="tiny">An overview of your archive, a monthly timeline, and its most recent items.</p>
          <button id="loadLibraryBtn" type="button">بارگذاری کتابخانه</button>
          <div class="status" id="libraryStatus"></div>
          <div class="tiny" id="libraryStats" style="margin-top:10px;"></div>
          <div class="tiny" id="libraryTimeline" style="margin-top:10px;"></div>
          <div class="results" id="libraryRecent"></div>
        </div>

        <div class="card reveal">
          <div class="card-head">
            <div class="card-icon">⬆️</div>
            <div>
              <h2>ورود بکاپ تلگرام</h2>
              <p class="sub">فایل JSON/ZIP خروجی Telegram Desktop</p>
            </div>
          </div>
          <p class="tiny">
            در Telegram Desktop مسیر «Export chat history» را با فرمت
            <b>Machine-readable JSON</b> بزن، سپس فایل <code>result.json</code> یا
            <code>.zip</code> آن را اینجا آپلود کن تا قابل جستجو شود و یک نسخه‌ی Markdown بگیری.
          </p>
          <label for="backupFile">Backup file (.json / .zip)</label>
          <input id="backupFile" type="file" accept=".json,.zip,application/json,application/zip" />
          <button id="importBackupBtn" type="button">ورود و تبدیل</button>
          <div class="status" id="backupStatus"></div>
          <div class="tiny" id="backupDownload" style="margin-top:10px;"></div>
        </div>

        <div class="card reveal">
          <div class="card-head">
            <div class="card-icon">🛠️</div>
            <div>
              <h2>جعبه‌ابزار تلگرام</h2>
              <p class="sub">روی حساب متصل سرور عمل می‌کند</p>
            </div>
          </div>
          <p class="tiny">Acts on the server's connected account (TELEGRAM_SESSION_STRING).</p>
          <div class="btn-row">
            <button id="accountBtn" type="button">حساب</button>
            <button id="deletedBtn" class="secondary" type="button">پیام‌های حذف‌شده</button>
          </div>
          <label for="toolsetPeer" style="margin-top:10px;">Chat (for scheduled / export)</label>
          <input id="toolsetPeer" placeholder="@channel or https://t.me/..." />
          <div class="btn-row">
            <button id="scheduledBtn" type="button">زمان‌بندی‌شده</button>
            <button id="llmExportBtn" class="secondary" type="button">خروجی LLM</button>
          </div>
          <label for="resendTarget" style="margin-top:10px;">Resend target</label>
          <input id="resendTarget" placeholder="@user or chat id" />
          <label for="resendText">Resend text</label>
          <textarea id="resendText" placeholder="Message to send on your behalf"></textarea>
          <button id="resendBtn" type="button">ارسال</button>
          <div class="status" id="toolsetStatus"></div>
          <div class="results" id="toolsetResults"></div>
        </div>
      </section>

      <div id="brainAnswer" style="display:none; margin-top:20px;" class="card">
        <h3>✦ پاسخ مغز هوش مصنوعی</h3>
        <p id="answerText" style="white-space: pre-wrap; color: var(--ink);"></p>
        <div class="tiny" id="answerMeta"></div>
      </div>

      <section class="results" id="results"></section>
    </main>

    <script>
      // Cinematic ambience: drifting golden embers + scroll reveal.
      (function () {
        var c = document.getElementById("fx");
        if (!c || !c.getContext) return;
        var ctx = c.getContext("2d"), W = 0, H = 0, dots = [];
        var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        function resize() { W = c.width = window.innerWidth; H = c.height = window.innerHeight; }
        function make() {
          dots = [];
          var n = Math.min(70, Math.floor(W / 22));
          for (var i = 0; i < n; i++) {
            dots.push({
              x: Math.random() * W, y: Math.random() * H, r: Math.random() * 1.6 + 0.4,
              s: Math.random() * 0.4 + 0.1, o: Math.random() * 0.5 + 0.2, t: Math.random() * Math.PI * 2
            });
          }
        }
        function tick() {
          ctx.clearRect(0, 0, W, H);
          for (var i = 0; i < dots.length; i++) {
            var d = dots[i];
            d.y -= d.s; d.t += 0.01; d.x += Math.sin(d.t) * 0.2;
            if (d.y < -10) { d.y = H + 10; d.x = Math.random() * W; }
            var a = d.o * (0.6 + 0.4 * Math.sin(d.t));
            ctx.beginPath(); ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(233,205,151," + a.toFixed(3) + ")"; ctx.fill();
          }
          requestAnimationFrame(tick);
        }
        window.addEventListener("resize", function () { resize(); make(); });
        resize(); make();
        if (!reduce) tick();
      })();
      (function () {
        var els = document.querySelectorAll(".reveal");
        if (!("IntersectionObserver" in window)) {
          els.forEach(function (e) { e.classList.add("in"); });
          return;
        }
        var io = new IntersectionObserver(function (entries) {
          entries.forEach(function (en) {
            if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); }
          });
        }, { threshold: 0.12 });
        els.forEach(function (e) { io.observe(e); });
      })();
    </script>
    <script>
      const ingestBtn = document.getElementById("ingestBtn");
      const searchBtn = document.getElementById("searchBtn");
      const askBtn = document.getElementById("askBtn");
      const reloadModelsBtn = document.getElementById("reloadModelsBtn");
      const saveSettingsBtn = document.getElementById("saveSettingsBtn");
      const ingestStatus = document.getElementById("ingestStatus");
      const searchStatus = document.getElementById("searchStatus");
      const settingsStatus = document.getElementById("settingsStatus");
      const settingsSummary = document.getElementById("settingsSummary");
      const results = document.getElementById("results");
      const brainAnswer = document.getElementById("brainAnswer");
      const answerText = document.getElementById("answerText");

      function displayResults(dataItems) {
        results.innerHTML = "";
        for (const item of dataItems) {
          const div = document.createElement("article");
          div.className = "result";
          div.innerHTML = `
            <div class="meta">
              <strong>${item.channel_title || item.channel_url}</strong>
              • ${item.media_kind}
              • score=${item.score}
              ${item.message_url ? `• <a href="${item.message_url}" target="_blank" rel="noreferrer">post</a>` : ""}
            </div>
            <div>${item.chunk_text}</div>
          `;
          results.appendChild(div);
        }
      }

      const transcriptionProvider = document.getElementById("transcriptionProvider");
      const transcriptionModel = document.getElementById("transcriptionModel");
      const embeddingProvider = document.getElementById("embeddingProvider");
      const embeddingModel = document.getElementById("embeddingModel");
      const geminiApiKey = document.getElementById("geminiApiKey");
      const openaiApiKey = document.getElementById("openaiApiKey");

      function fillSelect(select, models, selectedId) {
        select.innerHTML = "";
        for (const model of models) {
          const option = document.createElement("option");
          option.value = model.id;
          option.textContent = model.display_name || model.id;
          if (model.id === selectedId) {
            option.selected = true;
          }
          select.appendChild(option);
        }
        if (!models.length && selectedId) {
          const fallback = document.createElement("option");
          fallback.value = selectedId;
          fallback.textContent = selectedId;
          fallback.selected = true;
          select.appendChild(fallback);
        }
      }

      function withToken(options) {
        const opts = { ...(options || {}) };
        opts.headers = { ...(opts.headers || {}) };
        const token = localStorage.getItem("apiToken");
        if (token) {
          opts.headers["X-API-Token"] = token;
        }
        return opts;
      }

      async function fetchJson(url, options = undefined) {
        let response = await fetch(url, withToken(options));
        if (response.status === 401) {
          const token = window.prompt("This API requires a token (WEB_API_TOKEN). Enter it:");
          if (token) {
            localStorage.setItem("apiToken", token.trim());
            response = await fetch(url, withToken(options));
          }
        }
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "Request failed");
        }
        return data;
      }

      async function loadSettings() {
        const data = await fetchJson("/api/settings");
        transcriptionProvider.value = data.transcription_provider;
        embeddingProvider.value = data.embedding_provider;
        settingsSummary.textContent =
          `OpenAI key: ${data.openai_enabled ? "set" : "missing"} | Gemini key: ${data.gemini_enabled ? "set" : "missing"}`;
        await reloadModels();
        transcriptionModel.value = data.transcription_model;
        embeddingModel.value = data.embedding_model;
      }

      async function reloadModels() {
        settingsStatus.textContent = "در حال گرفتن لیست مدل‌ها...";
        try {
          const [transcriptionModels, embeddingModels] = await Promise.all([
            fetchJson(`/api/models?provider=${encodeURIComponent(transcriptionProvider.value)}&capability=transcription`),
            fetchJson(`/api/models?provider=${encodeURIComponent(embeddingProvider.value)}&capability=embedding`)
          ]);
          fillSelect(transcriptionModel, transcriptionModels.models, transcriptionModel.value);
          fillSelect(embeddingModel, embeddingModels.models, embeddingModel.value);
          settingsStatus.textContent = "لیست مدل‌ها بروزرسانی شد";
        } catch (error) {
          settingsStatus.textContent = error.message;
        }
      }

      transcriptionProvider.addEventListener("change", reloadModels);
      embeddingProvider.addEventListener("change", reloadModels);
      reloadModelsBtn.addEventListener("click", reloadModels);

      saveSettingsBtn.addEventListener("click", async () => {
        settingsStatus.textContent = "در حال ذخیره تنظیمات...";
        try {
          const data = await fetchJson("/api/settings", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              transcription_provider: transcriptionProvider.value,
              transcription_model: transcriptionModel.value,
              embedding_provider: embeddingProvider.value,
              embedding_model: embeddingModel.value,
              gemini_api_key: geminiApiKey.value || undefined,
              openai_api_key: openaiApiKey.value || undefined
            })
          });
          geminiApiKey.value = "";
          openaiApiKey.value = "";
          settingsSummary.textContent =
            `OpenAI key: ${data.openai_enabled ? "set" : "missing"} | Gemini key: ${data.gemini_enabled ? "set" : "missing"}`;
          settingsStatus.textContent = "تنظیمات ذخیره شد";
        } catch (error) {
          settingsStatus.textContent = error.message;
        }
      });

      ingestBtn.addEventListener("click", async () => {
        ingestStatus.textContent = "در حال ingest...";
        try {
          const data = await fetchJson("/api/channels/ingest", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              channel_url: document.getElementById("channelUrl").value,
              limit: Number(document.getElementById("limit").value || 50)
            })
          });
          ingestStatus.textContent = `کانال ${data.channel_title || data.channel_url} پردازش شد. media موفق: ${data.processed_media}`;
        } catch (error) {
          ingestStatus.textContent = error.message;
        }
      });

      searchBtn.addEventListener("click", async () => {
        searchStatus.textContent = "در حال جست‌وجو...";
        results.innerHTML = "";
        brainAnswer.style.display = "none";
        try {
          const data = await fetchJson("/api/search", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              query: document.getElementById("query").value,
              channel_url: document.getElementById("searchChannel").value || null,
              top_k: 8
            })
          });
          searchStatus.textContent = `${data.results.length} نتیجه پیدا شد`;
          displayResults(data.results);
        } catch (error) {
          searchStatus.textContent = error.message;
        }
      });

      askBtn.addEventListener("click", async () => {
        searchStatus.textContent = "در حال تفکر...";
        results.innerHTML = "";
        brainAnswer.style.display = "none";
        try {
          const data = await fetchJson("/api/ask", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              query: document.getElementById("query").value,
              channel_url: document.getElementById("searchChannel").value || null,
            })
          });
          searchStatus.textContent = "پاسخ آماده شد!";
          brainAnswer.style.display = "block";
          answerText.textContent = data.answer;
          const cited = new Set(data.cited || []);
          results.innerHTML = "";
          (data.sources || []).forEach((item, idx) => {
            const n = idx + 1;
            const div = document.createElement("article");
            div.className = "result";
            div.innerHTML = `
              <div class="meta">
                <strong>[${n}]${cited.has(n) ? " ✓" : ""}</strong>
                ${item.channel_title || item.channel_url}
                • ${item.media_kind}
                • score=${item.score}
                ${item.message_url ? `• <a href="${item.message_url}" target="_blank" rel="noreferrer">post</a>` : ""}
              </div>
              <div>${item.chunk_text}</div>
            `;
            results.appendChild(div);
          });
        } catch (error) {
          searchStatus.textContent = error.message;
        }
      });

      const loadLibraryBtn = document.getElementById("loadLibraryBtn");
      const libraryStatus = document.getElementById("libraryStatus");
      const libraryStats = document.getElementById("libraryStats");
      const libraryTimeline = document.getElementById("libraryTimeline");
      const libraryRecent = document.getElementById("libraryRecent");

      loadLibraryBtn.addEventListener("click", async () => {
        libraryStatus.textContent = "در حال بارگذاری...";
        libraryStats.textContent = "";
        libraryTimeline.textContent = "";
        libraryRecent.innerHTML = "";
        try {
          const [stats, timeline, recent] = await Promise.all([
            fetchJson("/api/stats"),
            fetchJson("/api/timeline?granularity=month"),
            fetchJson("/api/recent?limit=10"),
          ]);
          const kinds = Object.entries(stats.by_kind || {}).map(([k, v]) => `${k}: ${v}`).join(", ");
          libraryStats.textContent =
            `Items: ${stats.items} · Sources: ${stats.sources} · Tags: ${stats.tags}` +
            (kinds ? ` · ${kinds}` : "");
          const periods = (timeline.periods || []);
          if (periods.length) {
            libraryTimeline.textContent =
              "Timeline — " + periods.map(p => `${p.period}: ${p.count}`).join(" · ");
          }
          for (const item of (recent.items || [])) {
            const el = document.createElement("div");
            el.className = "result";
            const meta = document.createElement("div");
            meta.className = "meta";
            meta.textContent = `${item.source}${item.date ? " · " + item.date : ""}`;
            const body = document.createElement("div");
            body.textContent = item.snippet || "";
            el.appendChild(meta);
            el.appendChild(body);
            libraryRecent.appendChild(el);
          }
          libraryStatus.textContent = "";
        } catch (error) {
          libraryStatus.textContent = error.message;
        }
      });

      const backupFile = document.getElementById("backupFile");
      const importBackupBtn = document.getElementById("importBackupBtn");
      const backupStatus = document.getElementById("backupStatus");
      const backupDownload = document.getElementById("backupDownload");

      importBackupBtn.addEventListener("click", async () => {
        const file = backupFile.files && backupFile.files[0];
        if (!file) {
          backupStatus.textContent = "اول یک فایل .json یا .zip انتخاب کن.";
          return;
        }
        backupStatus.textContent = "در حال آپلود و پردازش... (ممکن است کمی طول بکشد)";
        backupDownload.innerHTML = "";
        try {
          const data = await fetchJson("/api/backup/import", {
            method: "POST",
            headers: {
              "content-type": "application/octet-stream",
              "X-Filename": file.name,
            },
            body: file,
          });
          backupStatus.textContent =
            `وارد شد: ${data.messages} پیام از ${data.chats} چت. حالا با Search/Ask قابل جستجوست.`;
          const blob = new Blob([data.markdown || ""], { type: "text/markdown" });
          const url = URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url;
          link.download = "telegram-backup.md";
          link.textContent = "دانلود فایل Markdown";
          backupDownload.appendChild(link);
        } catch (error) {
          backupStatus.textContent = error.message;
        }
      });

      // --- Telegram Toolset ---
      const toolsetStatus = document.getElementById("toolsetStatus");
      const toolsetResults = document.getElementById("toolsetResults");
      const toolsetPeer = document.getElementById("toolsetPeer");

      function renderToolset(lines) {
        toolsetResults.innerHTML = "";
        for (const line of lines) {
          const el = document.createElement("div");
          el.className = "result";
          el.textContent = line;
          toolsetResults.appendChild(el);
        }
      }

      async function runToolset(label, fn) {
        toolsetStatus.textContent = label + "…";
        toolsetResults.innerHTML = "";
        try {
          await fn();
          toolsetStatus.textContent = "";
        } catch (error) {
          toolsetStatus.textContent = error.message;
        }
      }

      document.getElementById("accountBtn").addEventListener("click", () => runToolset("Account", async () => {
        const a = await fetchJson("/api/account");
        const name = [a.first_name, a.last_name].filter(Boolean).join(" ") || "—";
        renderToolset([
          "Name: " + name,
          "Username: " + (a.username ? "@" + a.username : "—"),
          "ID: " + (a.id || "—"),
          "Phone: " + (a.phone || "—"),
          "Premium: " + (a.premium ? "yes" : "no"),
        ]);
      }));

      document.getElementById("deletedBtn").addEventListener("click", () => runToolset("Recovered deleted", async () => {
        const d = await fetchJson("/api/deleted?limit=20");
        const items = d.items || [];
        renderToolset(items.length ? items.map(r => `${r.sender || "Unknown"}${r.chat_title ? " · " + r.chat_title : ""}: ${r.text || ""}`) : ["No recovered deleted messages."]);
      }));

      document.getElementById("scheduledBtn").addEventListener("click", () => runToolset("Scheduled", async () => {
        const peer = toolsetPeer.value.trim();
        if (!peer) throw new Error("Enter a chat first.");
        const d = await fetchJson("/api/scheduled?peer=" + encodeURIComponent(peer));
        const items = d.items || [];
        renderToolset(items.length ? items.map(it => `#${it.id} · ${it.date || "?"} — ${it.text || "(media)"}`) : ["No scheduled messages."]);
      }));

      document.getElementById("llmExportBtn").addEventListener("click", () => runToolset("LLM export", async () => {
        const peer = toolsetPeer.value.trim();
        if (!peer) throw new Error("Enter a chat first.");
        const d = await fetchJson("/api/llmexport", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ peer, limit: 200 }) });
        const blob = new Blob([d.markdown || ""], { type: "text/markdown" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "telegram-" + (d.chat_label || "chat").replace(/[^a-z0-9_-]+/gi, "-").toLowerCase() + ".md";
        link.click();
        URL.revokeObjectURL(url);
        renderToolset(["Exported " + (d.count || 0) + " message(s) from " + (d.chat_label || peer) + "."]);
      }));

      document.getElementById("resendBtn").addEventListener("click", () => runToolset("Send", async () => {
        const target = document.getElementById("resendTarget").value.trim();
        const text = document.getElementById("resendText").value.trim();
        if (!target || !text) throw new Error("Target and text are required.");
        const d = await fetchJson("/api/resend", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ target, text }) });
        renderToolset(["Sent to " + target + " (message #" + (d.message_id || "?") + ")."]);
      }));

      loadSettings().catch(error => {
        settingsStatus.textContent = error.message;
      });
    </script>
  </body>
</html>
"""


LANDING_HTML = """
<!doctype html>
<html lang="fa" dir="rtl">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Telegram Notebook — حافظه‌ی هوشمند تلگرام شما</title>
    <meta name="description" content="آرشیو کانال‌ها، چت‌ها و بکاپ‌های تلگرام را به یک حافظه‌ی قابل‌جستجو و قابل‌اتصال به ابزارهای AI تبدیل کن." />
    <meta name="theme-color" content="#06070b" />
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500&family=Inter:wght@400;500;600&family=Vazirmatn:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
    <style>
      :root {
        --bg-0: #06070b;
        --ink: #f3efe6;
        --muted: #98a1b2;
        --gold: #e9cd97;
        --gold-soft: #f3ddae;
        --emerald: #34d8a8;
        --teal: #14b894;
        --line: rgba(233, 205, 151, 0.16);
        --line-soft: rgba(255, 255, 255, 0.07);
        --card: rgba(15, 20, 30, 0.62);
      }
      * { box-sizing: border-box; }
      html { scroll-behavior: smooth; }
      body {
        margin: 0; min-height: 100vh; color: var(--ink); background: var(--bg-0);
        overflow-x: hidden; -webkit-font-smoothing: antialiased;
        font-family: "Vazirmatn", "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
      }
      ::selection { background: rgba(233, 205, 151, 0.28); color: #fff; }
      a { color: var(--gold); text-decoration: none; }
      code { background: rgba(255, 255, 255, 0.06); padding: 1px 6px; border-radius: 6px; font-size: 0.85em; }

      /* ---------- cinematic background ---------- */
      .bg { position: fixed; inset: 0; z-index: -1; overflow: hidden; }
      .bg-aurora {
        position: absolute; inset: -25%;
        background:
          radial-gradient(40% 40% at 18% 10%, rgba(20, 184, 148, 0.22), transparent 60%),
          radial-gradient(40% 40% at 82% 14%, rgba(233, 205, 151, 0.20), transparent 60%),
          radial-gradient(55% 55% at 50% 108%, rgba(36, 80, 120, 0.30), transparent 65%),
          linear-gradient(180deg, #06070b 0%, #080b12 45%, #06070b 100%);
        animation: drift 28s ease-in-out infinite alternate;
      }
      @keyframes drift { 0% { transform: translate3d(0,0,0) scale(1); } 100% { transform: translate3d(0,-3%,0) scale(1.07); } }
      .bg-grid {
        position: absolute; inset: 0; opacity: 0.5;
        background-image:
          linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px);
        background-size: 64px 64px;
        -webkit-mask-image: radial-gradient(circle at 50% 22%, #000 0%, transparent 72%);
        mask-image: radial-gradient(circle at 50% 22%, #000 0%, transparent 72%);
      }
      .bg-grain {
        position: absolute; inset: 0; pointer-events: none; opacity: 0.05; mix-blend-mode: overlay;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
      }
      .bg-vignette { position: absolute; inset: 0; pointer-events: none; background: radial-gradient(120% 90% at 50% 0%, transparent 55%, rgba(0, 0, 0, 0.6) 100%); }

      .wrap { max-width: 1080px; margin: 0 auto; padding: 0 20px 84px; }
      nav {
        position: sticky; top: 0; z-index: 20;
        display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;
        padding: 16px 0; margin-bottom: 8px; backdrop-filter: blur(14px);
      }
      .brand { font-weight: 700; font-size: 1.2rem; letter-spacing: -0.01em; }
      .nav-links { display: flex; gap: 18px; align-items: center; flex-wrap: wrap; }
      .nav-links a { color: var(--muted); transition: color 0.2s; }
      .nav-links a:hover { color: var(--ink); }
      .btn {
        display: inline-block; padding: 12px 24px; border-radius: 999px; font: inherit; font-weight: 600; cursor: pointer;
        color: #06120e; border: none; background: linear-gradient(120deg, var(--emerald), var(--teal));
        box-shadow: 0 12px 34px rgba(20, 184, 148, 0.3); transition: transform 0.2s, box-shadow 0.2s, filter 0.2s;
      }
      .btn:hover { transform: translateY(-2px); filter: brightness(1.06); box-shadow: 0 18px 44px rgba(20, 184, 148, 0.42); }
      .btn.secondary { color: #1a1206; background: linear-gradient(120deg, var(--gold-soft), var(--gold)); box-shadow: 0 12px 34px rgba(233, 205, 151, 0.28); }
      .btn.ghost { background: transparent; color: var(--ink); border: 1px solid var(--line); box-shadow: none; }
      .btn.ghost:hover { border-color: var(--gold); color: var(--gold); }

      .hero { padding: 62px 0 16px; }
      .eyebrow {
        display: inline-block; padding: 7px 17px; border-radius: 999px; margin-bottom: 22px;
        font-size: 0.8rem; letter-spacing: 0.6px; color: var(--gold-soft);
        border: 1px solid var(--line); background: rgba(233, 205, 151, 0.06);
      }
      .hero h1 { margin: 0 0 14px; font-weight: 600; font-size: clamp(2.2rem, 6vw, 4.2rem); line-height: 1.16; letter-spacing: -0.01em; max-width: 20ch; }
      .grad { background: linear-gradient(100deg, var(--emerald), var(--gold) 58%, var(--gold-soft)); -webkit-background-clip: text; background-clip: text; color: transparent; }
      .hero p { color: var(--muted); font-size: 1.12rem; line-height: 2; max-width: 60ch; }
      .cta { display: flex; gap: 12px; margin-top: 26px; flex-wrap: wrap; }

      section.block { margin-top: 66px; }
      h2 { font-size: clamp(1.6rem, 3vw, 2.3rem); letter-spacing: -0.01em; margin: 0 0 22px; font-weight: 600; }
      .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
      .card {
        position: relative; overflow: hidden;
        backdrop-filter: blur(18px); background: var(--card);
        border: 1px solid var(--line-soft); border-radius: 20px; padding: 22px;
        box-shadow: 0 30px 80px rgba(0, 0, 0, 0.5); transition: transform 0.4s cubic-bezier(.2, .7, .2, 1), border-color 0.4s;
      }
      .card::before {
        content: ""; position: absolute; inset: 0; border-radius: inherit; padding: 1px;
        background: linear-gradient(140deg, rgba(233, 205, 151, 0.45), transparent 42%, rgba(52, 216, 168, 0.3));
        -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
        -webkit-mask-composite: xor; mask-composite: exclude; opacity: 0; transition: opacity 0.4s; pointer-events: none;
      }
      .card:hover { transform: translateY(-4px); border-color: transparent; }
      .card:hover::before { opacity: 1; }
      .card h3 { margin: 0 0 8px; font-size: 1.18rem; }
      .card p { color: var(--muted); line-height: 1.9; margin: 0; }

      .steps { display: grid; gap: 14px; }
      .step {
        display: flex; gap: 16px; align-items: flex-start; padding: 18px 20px;
        background: var(--card); border: 1px solid var(--line-soft); border-radius: 18px; backdrop-filter: blur(12px);
      }
      .step .num {
        flex: 0 0 auto; width: 40px; height: 40px; border-radius: 50%;
        display: grid; place-items: center; font-weight: 700; color: #06120e;
        background: linear-gradient(135deg, var(--emerald), var(--teal));
      }
      .step .num.two { color: #1a1206; background: linear-gradient(135deg, var(--gold-soft), var(--gold)); }
      .step b { color: var(--ink); }

      .panel {
        position: relative; overflow: hidden;
        background: var(--card); border: 1px solid var(--line-soft); border-radius: 22px;
        padding: 30px; box-shadow: 0 30px 80px rgba(0, 0, 0, 0.5); backdrop-filter: blur(18px);
      }
      footer { margin-top: 74px; padding-top: 24px; border-top: 1px solid var(--line-soft); color: var(--muted); }
      .muted { color: var(--muted); }

      .rise { opacity: 0; animation: rise 0.9s cubic-bezier(.2, .7, .2, 1) forwards; }
      @keyframes rise { from { opacity: 0; transform: translateY(22px); } to { opacity: 1; transform: none; } }

      ::-webkit-scrollbar { width: 11px; }
      ::-webkit-scrollbar-track { background: #06070b; }
      ::-webkit-scrollbar-thumb { background: linear-gradient(var(--teal), var(--gold)); border-radius: 99px; border: 3px solid #06070b; }
    </style>
  </head>
  <body>
    <div class="bg" aria-hidden="true">
      <div class="bg-aurora"></div>
      <div class="bg-grid"></div>
      <div class="bg-grain"></div>
      <div class="bg-vignette"></div>
    </div>

    <div class="wrap">
      <nav>
        <div class="brand">📓 Telegram Notebook</div>
        <div class="nav-links">
          <a href="#features">امکانات</a>
          <a href="#how">چطور کار می‌کند</a>
          <a href="#backup">Import بکاپ</a>
          <a class="btn ghost" href="/app">ورود به اپ</a>
        </div>
      </nav>

      <header class="hero rise">
        <span class="eyebrow">NotebookLM برای تلگرام</span>
        <h1>تلگرام شما، به یک <span class="grad">حافظه‌ی هوشمند</span> تبدیل می‌شود.</h1>
        <p>
          کانال‌ها، چت‌ها، پیام‌های فورواردشده و حتی <b>فایل بکاپ تلگرام</b> را وارد کن؛
          همه به متن قابل‌جستجو تبدیل می‌شوند و می‌توانی مثل یک NotebookLM شخصی از آرشیو
          خودت سؤال بپرسی — هم از داخل ربات تلگرام، هم از همین وب‌سایت.
        </p>
        <div class="cta">
          <a class="btn" href="/app">شروع از وب‌اپ</a>
          <a class="btn secondary" href="#backup">Import فایل بکاپ</a>
        </div>
      </header>

      <section id="features" class="block">
        <h2>چه کاری انجام می‌دهد؟</h2>
        <div class="grid">
          <div class="card">
            <h3>🔎 جستجوی معنایی</h3>
            <p>جستجوی keyword و semantic روی کل آرشیو؛ نتایج همراه با منبع و لینک پیام.</p>
          </div>
          <div class="card">
            <h3>🧠 پرسش و پاسخ (RAG)</h3>
            <p>مثل NotebookLM از آرشیو خودت سؤال بپرس و پاسخ مستند بگیر.</p>
          </div>
          <div class="card">
            <h3>🗂️ Import بکاپ تلگرام</h3>
            <p>فایل JSON/ZIP خروجی Telegram Desktop را بده تا قابل‌جستجو شود و Markdown بگیری.</p>
          </div>
          <div class="card">
            <h3>🏷️ تگ و دفترچه</h3>
            <p>قوانین keyword/AI برای تگ‌گذاری خودکار و گروه‌بندی تگ‌ها زیر یک collection.</p>
          </div>
          <div class="card">
            <h3>📨 Forwarded Inbox</h3>
            <p>هر پیام/فایل را به ربات فوروارد کن تا ذخیره، پردازش و قابل‌جستجو شود.</p>
          </div>
          <div class="card">
            <h3>🔌 MCP برای ابزارهای AI</h3>
            <p>آرشیو را به‌صورت read-only به Claude، Cursor و سایر AI clientها وصل کن.</p>
          </div>
        </div>
      </section>

      <section id="how" class="block">
        <h2>چطور کار می‌کند</h2>
        <div class="steps">
          <div class="step">
            <div class="num">۱</div>
            <div><b>وارد کن.</b> یک کانال را ingest کن، پیام‌ها را فوروارد کن، یا فایل بکاپ تلگرام را آپلود کن.</div>
          </div>
          <div class="step">
            <div class="num two">۲</div>
            <div><b>پردازش.</b> محتوا به متن تبدیل، chunk و ایندکس می‌شود و با قوانین تو تگ می‌خورد.</div>
          </div>
          <div class="step">
            <div class="num">۳</div>
            <div><b>بپرس.</b> از وب‌اپ یا ربات تلگرام جستجو کن، سؤال بپرس، خلاصه و timeline بگیر.</div>
          </div>
        </div>
      </section>

      <section id="backup" class="block">
        <h2>Import فایل بکاپ تلگرام</h2>
        <div class="panel">
          <p class="muted" style="line-height:1.9;">
            در <b>Telegram Desktop</b> به <b>Settings → Advanced → Export Telegram data</b>
            (یا روی یک چت: <b>Export chat history</b>) برو و فرمت را روی
            <b>Machine-readable JSON</b> بگذار. سپس فایل <code>result.json</code> یا
            <code>.zip</code> خروجی را در وب‌اپ آپلود کن. تمام پیام‌ها قابل‌جستجو می‌شوند و
            یک نسخه‌ی <b>Markdown</b> تحویل می‌گیری.
          </p>
          <div class="cta">
            <a class="btn" href="/app">باز کردن وب‌اپ و آپلود بکاپ</a>
          </div>
        </div>
      </section>

      <footer>
        <p>
          ساخته‌شده برای تبدیل تلگرام به «حافظه‌ای برای دستیارهای AI».
          <a href="/app">ورود به اپ</a> ·
          <a href="https://github.com/shm379/telegram-notebooklm-mvp">سورس روی GitHub</a>
        </p>
      </footer>
    </div>
  </body>
</html>
"""


MINIAPP_HTML = """
<!doctype html>
<html lang="fa" dir="rtl">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
    <title>Telegram Notebook — Mini App</title>
    <meta name="theme-color" content="#06070b" />
    <script src="https://telegram.org/js/telegram-web-app.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Vazirmatn:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
    <style>
      :root {
        --ink: #f3efe6; --muted: #98a1b2; --gold: #e9cd97; --gold-soft: #f3ddae;
        --emerald: #34d8a8; --teal: #14b894;
        --line: rgba(255,255,255,0.08); --card: rgba(15,20,30,0.66);
      }
      * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
      html, body { margin: 0; }
      body {
        min-height: 100vh; color: var(--ink); background: #06070b;
        font-family: "Vazirmatn", "Inter", system-ui, -apple-system, sans-serif;
        -webkit-font-smoothing: antialiased; overflow-x: hidden;
        padding: 16px 14px calc(28px + env(safe-area-inset-bottom));
      }
      .bg { position: fixed; inset: 0; z-index: -1; overflow: hidden; }
      .bg::before {
        content: ""; position: absolute; inset: -25%;
        background:
          radial-gradient(40% 36% at 20% 8%, rgba(20,184,148,0.24), transparent 60%),
          radial-gradient(42% 40% at 84% 12%, rgba(233,205,151,0.20), transparent 60%),
          radial-gradient(60% 50% at 50% 110%, rgba(36,80,120,0.30), transparent 65%),
          linear-gradient(180deg, #06070b, #080b12 50%, #06070b);
        animation: drift 26s ease-in-out infinite alternate;
      }
      @keyframes drift { 0% { transform: translateY(0) scale(1); } 100% { transform: translateY(-3%) scale(1.06); } }

      .head { display: flex; align-items: center; gap: 12px; margin: 6px 2px 18px; }
      .avatar {
        width: 46px; height: 46px; border-radius: 14px; display: grid; place-items: center;
        font-size: 1.2rem; font-weight: 700; color: #06120e;
        background: linear-gradient(135deg, var(--emerald), var(--gold-soft));
        box-shadow: 0 0 26px rgba(52,216,168,0.3);
      }
      .head .hi { font-size: 0.82rem; color: var(--muted); }
      .head .name { font-size: 1.12rem; font-weight: 600; }
      .head .name em { font-style: normal; color: var(--gold); }

      .chips { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
      .chip {
        flex: 1; min-width: 90px; padding: 12px; border-radius: 14px; text-align: center;
        background: var(--card); border: 1px solid var(--line); backdrop-filter: blur(8px);
      }
      .chip b { display: block; font-size: 1.2rem; color: var(--gold); }
      .chip span { font-size: 0.72rem; color: var(--muted); }

      .card {
        background: var(--card); border: 1px solid var(--line); border-radius: 20px;
        padding: 16px; margin-bottom: 14px; backdrop-filter: blur(16px);
        box-shadow: 0 24px 60px rgba(0,0,0,0.5);
      }
      .card h2 { margin: 0 0 12px; font-size: 1rem; font-weight: 600; }
      textarea {
        width: 100%; min-height: 92px; resize: vertical; border-radius: 14px; padding: 13px;
        font: inherit; line-height: 1.8; color: var(--ink);
        background: rgba(255,255,255,0.04); border: 1px solid var(--line);
      }
      textarea:focus { outline: none; border-color: var(--gold); box-shadow: 0 0 0 3px rgba(233,205,151,0.13); }
      textarea::placeholder { color: rgba(152,161,178,0.6); }
      button {
        width: 100%; margin-top: 12px; padding: 14px; border: none; border-radius: 14px; cursor: pointer;
        font: inherit; font-weight: 600; color: #06120e; letter-spacing: 0.2px;
        background: linear-gradient(120deg, var(--emerald), var(--teal));
        box-shadow: 0 12px 30px rgba(20,184,148,0.32); transition: transform .15s, filter .15s;
      }
      button:active { transform: translateY(1px); filter: brightness(0.97); }
      button[disabled] { opacity: 0.6; }

      .status { min-height: 20px; margin-top: 10px; font-size: 0.86rem; color: var(--emerald); }
      .answer { white-space: pre-wrap; line-height: 1.95; }
      .sources { margin-top: 12px; display: grid; gap: 8px; }
      .src { padding: 12px; border-radius: 13px; background: rgba(255,255,255,0.03); border: 1px solid var(--line); font-size: 0.86rem; line-height: 1.7; }
      .src .meta { color: var(--muted); font-size: 0.76rem; margin-bottom: 5px; }
      .src .meta b { color: var(--gold); }
      a { color: var(--gold); }
      .hint { color: var(--muted); font-size: 0.82rem; line-height: 1.8; text-align: center; }
      .hidden { display: none; }
    </style>
  </head>
  <body>
    <div class="bg" aria-hidden="true"></div>

    <div class="head">
      <div class="avatar" id="avatar">◎</div>
      <div>
        <div class="hi">حافظه‌ی هوشمند تلگرام</div>
        <div class="name" id="hello">Telegram <em>Notebook</em></div>
      </div>
    </div>

    <div class="chips" id="chips"></div>

    <div class="card">
      <h2>از آرشیو خودت بپرس</h2>
      <textarea id="q" placeholder="مثلاً: بین چیزهایی که ذخیره کردم، کدام‌ها درباره‌ی فلان موضوع بودند؟"></textarea>
      <button id="askBtn">پرسش از مغز AI</button>
      <div class="status" id="status"></div>
    </div>

    <div class="card hidden" id="answerCard">
      <h2>✦ پاسخ</h2>
      <div class="answer" id="answer"></div>
      <div class="sources" id="sources"></div>
    </div>

    <script>
      var tg = window.Telegram && window.Telegram.WebApp ? window.Telegram.WebApp : null;
      var initData = tg ? tg.initData : "";
      if (tg) {
        tg.ready(); tg.expand();
        try { tg.setHeaderColor && tg.setHeaderColor("#06070b"); } catch (e) {}
        var u = tg.initDataUnsafe && tg.initDataUnsafe.user;
        if (u) {
          var nm = (u.first_name || "") + (u.last_name ? " " + u.last_name : "");
          if (nm.trim()) document.getElementById("hello").textContent = nm.trim();
          var initials = (u.first_name || "?").trim().charAt(0).toUpperCase();
          document.getElementById("avatar").textContent = initials || "◎";
        }
      }

      function api(path, opts) {
        opts = opts || {};
        opts.headers = Object.assign({}, opts.headers, { "X-Telegram-Init-Data": initData });
        return fetch(path, opts).then(function (r) {
          return r.json().then(function (d) {
            if (!r.ok) throw new Error(d.detail || "Request failed");
            return d;
          });
        });
      }

      var statusEl = document.getElementById("status");
      var answerCard = document.getElementById("answerCard");
      var answerEl = document.getElementById("answer");
      var sourcesEl = document.getElementById("sources");
      var askBtn = document.getElementById("askBtn");

      function loadStats() {
        api("/api/miniapp/stats").then(function (s) {
          var k = s.by_kind || {};
          var media = Object.keys(k).reduce(function (a, key) { return a + (k[key] || 0); }, 0);
          var chips = [
            ["items", s.items || 0, "آیتم"],
            ["sources", s.sources || 0, "منبع"],
            ["tags", s.tags || 0, "تگ"],
          ];
          document.getElementById("chips").innerHTML = chips.map(function (c) {
            return '<div class="chip"><b>' + c[1] + '</b><span>' + c[2] + '</span></div>';
          }).join("");
        }).catch(function (e) {
          document.getElementById("chips").innerHTML = '<div class="hint">' +
            (initData ? e.message : "این صفحه را از داخل ربات تلگرام و با دکمه‌ی Mini App باز کن.") + '</div>';
        });
      }

      askBtn.addEventListener("click", function () {
        var q = document.getElementById("q").value.trim();
        if (!q) { statusEl.textContent = "اول سؤالت را بنویس."; return; }
        statusEl.textContent = "در حال تفکر…";
        askBtn.disabled = true; answerCard.classList.add("hidden");
        api("/api/miniapp/ask", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ query: q }),
        }).then(function (d) {
          statusEl.textContent = "";
          answerEl.textContent = d.answer || "";
          var cited = {}; (d.cited || []).forEach(function (n) { cited[n] = true; });
          sourcesEl.innerHTML = (d.sources || []).map(function (item, i) {
            var n = i + 1;
            var link = item.message_url ? ' · <a href="' + item.message_url + '" target="_blank" rel="noreferrer">post</a>' : "";
            return '<div class="src"><div class="meta"><b>[' + n + ']' + (cited[n] ? " ✓" : "") + '</b> ' +
              (item.channel_title || item.channel_url || "") + " · " + (item.media_kind || "text") + link +
              '</div>' + (item.chunk_text || "") + '</div>';
          }).join("");
          answerCard.classList.remove("hidden");
          if (tg) { try { tg.HapticFeedback.notificationOccurred("success"); } catch (e) {} }
        }).catch(function (e) {
          statusEl.textContent = e.message;
        }).then(function () { askBtn.disabled = false; });
      });

      loadStats();
    </script>
  </body>
</html>
"""


def _query_int(query: dict, name: str, *, default: int, lo: int, hi: int) -> int:
    """Read a clamped integer query param, falling back to ``default`` on bad input."""
    try:
        value = int(query.get(name, [str(default)])[0])
    except (ValueError, TypeError):
        value = default
    return max(lo, min(hi, value))


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "TelegramNotebook/0.3"

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        return json.loads(body.decode("utf-8"))

    def _send_json(self, payload: dict[str, object], status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, html: str, status: int = 200) -> None:
        raw = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _client_is_loopback(self) -> bool:
        host = self.client_address[0] if self.client_address else ""
        return host in {"127.0.0.1", "::1", "::ffff:127.0.0.1"}

    def _presented_token(self) -> str | None:
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[len("Bearer "):].strip()
        token = self.headers.get("X-API-Token")
        return token.strip() if token else None

    def _require_auth(self) -> bool:
        """Guard for /api endpoints. Returns True if the request may proceed.

        When WEB_API_TOKEN is configured, a matching bearer/X-API-Token is required.
        When it is not configured, only loopback clients are allowed so the API is not
        exposed unauthenticated over the network. Sends a 401 and returns False on failure.
        """
        configured = state.settings.web_api_token
        if configured:
            presented = self._presented_token()
            if presented and hmac.compare_digest(presented, configured):
                return True
            self._send_json({"detail": "Unauthorized"}, status=HTTPStatus.UNAUTHORIZED)
            return False
        if self._client_is_loopback():
            return True
        self._send_json(
            {"detail": "Unauthorized: set WEB_API_TOKEN to allow non-local access to the API."},
            status=HTTPStatus.UNAUTHORIZED,
        )
        return False

    def _miniapp_user_id(self) -> int | None:
        """Verify the Telegram Mini App initData header and return the user's id."""
        init_data = self.headers.get("X-Telegram-Init-Data", "")
        token = state.settings.telegram_bot_token or ""
        return init_data_user_id(verify_init_data(init_data, token))

    def _handle_miniapp_post(self, parsed) -> None:
        """Mini App search / ask, authenticated by initData and scoped to the user."""
        uid = self._miniapp_user_id()
        if uid is None:
            self._send_json({"detail": "Mini App authentication required"}, status=HTTPStatus.UNAUTHORIZED)
            return
        try:
            payload = self._read_json()
        except json.JSONDecodeError:
            self._send_json({"detail": "Invalid JSON payload"}, status=400)
            return
        query = str(payload.get("query", "")).strip()
        if not query:
            self._send_json({"detail": "query is required"}, status=400)
            return
        try:
            results = state.search_service.search(owner_id=uid, query=query, top_k=5)
            if parsed.path == "/api/miniapp/search":
                self._send_json({"query": query, "results": [r.to_dict() for r in results]})
                return
            answer = state.search_service.generate_answer(query=query, results=results, **state.llm_params())
            from . import citations
            self._send_json(
                {
                    "query": query,
                    "answer": answer,
                    "sources": [r.to_dict() for r in results],
                    "cited": citations.cited_indices(answer, len(results)),
                }
            )
        except Exception as exc:
            logger.exception("Mini App request failed")
            self._send_json({"detail": str(exc)}, status=400)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_html(LANDING_HTML)
                return
            if parsed.path in ("/app", "/app/"):
                self._send_html(INDEX_HTML)
                return
            if parsed.path in ("/miniapp", "/miniapp/"):
                self._send_html(MINIAPP_HTML)
                return
            if parsed.path == "/api/miniapp/stats":
                uid = self._miniapp_user_id()
                if uid is None:
                    self._send_json({"detail": "Mini App authentication required"}, status=HTTPStatus.UNAUTHORIZED)
                    return
                self._send_json(state.repository.archive_stats(owner_id=uid))
                return
            if parsed.path == "/api/health":
                config = state.runtime_config()
                self._send_json(
                    {
                        "ok": True,
                        **asdict(config),
                    }
                )
                return
            if parsed.path == "/api/settings":
                if not self._require_auth():
                    return
                self._send_json(asdict(state.runtime_config()))
                return
            if parsed.path == "/api/models":
                if not self._require_auth():
                    return
                query = parse_qs(parsed.query)
                provider = (query.get("provider", ["gemini"])[0] or "gemini").lower()
                capability = query.get("capability", [None])[0]
                models = state.list_models(provider=provider, capability=capability)
                self._send_json(
                    {
                        "provider": provider,
                        "capability": capability,
                        "models": models,
                    }
                )
                return
            if parsed.path == "/api/stats":
                if not self._require_auth():
                    return
                self._send_json(state.repository.archive_stats(owner_id=WEB_OWNER_ID))
                return
            if parsed.path == "/api/recent":
                if not self._require_auth():
                    return
                limit = _query_int(parse_qs(parsed.query), "limit", default=10, lo=1, hi=50)
                items = state.repository.timeline_items(owner_id=WEB_OWNER_ID, limit=limit)
                self._send_json({"items": recent_rows(items, limit=limit)})
                return
            if parsed.path == "/api/timeline":
                if not self._require_auth():
                    return
                query = parse_qs(parsed.query)
                granularity = "day" if query.get("granularity", ["month"])[0] == "day" else "month"
                items = state.repository.timeline_items(owner_id=WEB_OWNER_ID)
                self._send_json({"granularity": granularity, "periods": build_timeline(items, granularity=granularity)})
                return
            if parsed.path == "/api/account":
                if not self._require_auth():
                    return
                from . import toolset
                self._send_json(_run_account_op(lambda c: toolset.account_info(c)))
                return
            if parsed.path == "/api/scheduled":
                if not self._require_auth():
                    return
                peer = parse_qs(parsed.query).get("peer", [""])[0].strip()
                if not peer:
                    self._send_json({"detail": "peer is required"}, status=400)
                    return
                from . import toolset
                items = _run_account_op(lambda c: toolset.list_scheduled(c, peer))
                self._send_json({"peer": peer, "items": items})
                return
            if parsed.path == "/api/deleted":
                if not self._require_auth():
                    return
                limit = _query_int(parse_qs(parsed.query), "limit", default=20, lo=1, hi=100)
                self._send_json({"items": state.repository.list_recovered(owner_id=WEB_OWNER_ID, limit=limit)})
                return
            self._send_json({"detail": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            logger.exception("GET %s failed", parsed.path)
            self._send_json({"detail": str(exc)}, status=400)

    def _handle_backup_import(self, parsed) -> None:
        """Ingest an uploaded Telegram backup (raw .json/.zip body) and return Markdown.

        The file arrives as the raw request body with its name in ``X-Filename``
        (or ``?filename=``), so there is no multipart parsing to do. The backup is
        ingested into the shared web archive (``WEB_OWNER_ID``) and the response
        carries import counts plus the Markdown rendering for the client to save.
        """
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        if not body:
            self._send_json({"detail": "Empty upload"}, status=400)
            return
        filename = self.headers.get("X-Filename") or parse_qs(parsed.query).get("filename", [""])[0]
        try:
            resolver = make_zip_file_resolver(body)
            media_extractor = build_media_text_extractor(
                extraction=state.extraction, transcription=state.transcription
            ) if resolver else None
            chats = parse_export(
                read_export(body, filename), file_resolver=resolver, media_extractor=media_extractor
            )
        except Exception as exc:
            self._send_json({"detail": f"Could not read backup: {exc}"}, status=400)
            return
        try:
            result = asyncio.run(state.pipeline.ingest_backup(owner_id=WEB_OWNER_ID, chats=chats))
        except Exception as exc:
            logger.exception("Backup import failed")
            self._send_json({"detail": str(exc)}, status=400)
            return
        self._send_json(
            {
                "ok": True,
                "chats": result["chats"],
                "messages": result["messages"],
                "total_messages": count_messages(chats),
                "markdown": render_markdown(chats),
            }
        )

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        # Mini App endpoints authenticate via Telegram initData, not the
        # loopback/WEB_API_TOKEN gate, so they are handled before _require_auth.
        if parsed.path in ("/api/miniapp/ask", "/api/miniapp/search"):
            self._handle_miniapp_post(parsed)
            return

        if not self._require_auth():
            return

        if parsed.path == "/api/backup/import":
            try:
                self._handle_backup_import(parsed)
            except Exception as exc:
                logger.exception("Backup import handler failed")
                self._send_json({"detail": str(exc)}, status=400)
            return

        try:
            payload = self._read_json()
        except json.JSONDecodeError:
            self._send_json({"detail": "Invalid JSON payload"}, status=400)
            return

        if parsed.path == "/api/settings":
            updates: dict[str, str | None] = {}
            for key in (
                "transcription_provider",
                "transcription_model",
                "embedding_provider",
                "embedding_model",
                "llm_provider",
                "llm_model",
                "ollama_base_url",
                "gemini_api_key",
                "openai_api_key",
            ):
                if key in payload:
                    value = str(payload[key]).strip() if payload[key] is not None else ""
                    updates[key.upper()] = value
            if updates:
                upsert_env_values(updates)
                state.reload()
            self._send_json(asdict(state.runtime_config()))
            return

        if parsed.path == "/api/channels/ingest":
            channel_url = str(payload.get("channel_url", "")).strip()
            limit = int(payload.get("limit", 50))
            if not channel_url:
                self._send_json({"detail": "channel_url is required"}, status=400)
                return
            try:
                stats = asyncio.run(
                    state.pipeline.ingest_channel(owner_id=WEB_OWNER_ID, channel_url=channel_url, limit=limit)
                )
                self._send_json(
                    {
                        "channel_url": stats.channel_url,
                        "channel_title": stats.channel_title,
                        "processed_messages": stats.processed_messages,
                        "processed_media": stats.processed_media,
                        "skipped_media": stats.skipped_media,
                    }
                )
            except Exception as exc:
                logger.exception("Ingest request failed")
                self._send_json({"detail": str(exc)}, status=400)
            return

        if parsed.path == "/api/search":
            query = str(payload.get("query", "")).strip()
            channel_url = payload.get("channel_url")
            top_k = int(payload.get("top_k", state.settings.default_result_limit))
            if not query:
                self._send_json({"detail": "query is required"}, status=400)
                return
            try:
                results = state.search_service.search(
                    owner_id=WEB_OWNER_ID,
                    query=query,
                    channel_url=str(channel_url).strip() if channel_url else None,
                    top_k=top_k,
                    vertex_config=state.vertex_search_config(),
                )
                self._send_json(
                    {
                        "query": query,
                        "results": [result.to_dict() for result in results],
                    }
                )
            except Exception as exc:
                logger.exception("Search request failed")
                self._send_json({"detail": str(exc)}, status=400)
            return

        if parsed.path == "/api/ask":
            query = str(payload.get("query", "")).strip()
            channel_url = payload.get("channel_url")
            if not query:
                self._send_json({"detail": "query is required"}, status=400)
                return
            try:
                vertex_config = state.vertex_search_config()
                # 1. Search for relevant chunks
                results = state.search_service.search(
                    owner_id=WEB_OWNER_ID,
                    query=query,
                    channel_url=str(channel_url).strip() if channel_url else None,
                    top_k=5,
                    vertex_config=vertex_config,
                )
                # 2. Generate answer with the configured chat provider (default: Ollama)
                answer = state.search_service.generate_answer(
                    query=query,
                    results=results,
                    **state.llm_params(),
                )
                from . import citations
                self._send_json(
                    {
                        "query": query,
                        "answer": answer,
                        "sources": [result.to_dict() for result in results],
                        "cited": citations.cited_indices(answer, len(results)),
                    }
                )
            except Exception as exc:
                logger.exception("Ask request failed")
                self._send_json({"detail": str(exc)}, status=400)
            return

        if parsed.path == "/api/llmexport":
            peer = str(payload.get("peer", "")).strip()
            limit = int(payload.get("limit", 200))
            if not peer:
                self._send_json({"detail": "peer is required"}, status=400)
                return
            try:
                from . import toolset
                chat_label, rows = _run_account_op(lambda c: toolset.fetch_chat_messages(c, peer, limit=limit))
                self._send_json({
                    "peer": peer,
                    "chat_label": chat_label,
                    "count": len(rows),
                    "markdown": toolset.build_llm_export(chat_label, rows),
                })
            except Exception as exc:
                logger.exception("llmexport request failed")
                self._send_json({"detail": str(exc)}, status=400)
            return

        if parsed.path == "/api/resend":
            target = str(payload.get("target", "")).strip()
            text = str(payload.get("text", "")).strip()
            if not target or not text:
                self._send_json({"detail": "target and text are required"}, status=400)
                return
            try:
                from . import toolset
                msg_id = _run_account_op(lambda c: toolset.resend_text(c, target, text))
                self._send_json({"target": target, "message_id": msg_id})
            except Exception as exc:
                logger.exception("resend request failed")
                self._send_json({"detail": str(exc)}, status=400)
            return

        self._send_json({"detail": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        return


def _maybe_start_web_watcher() -> None:
    """Start a deleted-message watcher for the server account when WEB_WATCH_DELETED=1."""
    flag = (os.environ.get("WEB_WATCH_DELETED", "") or "").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return
    settings = state.settings
    if not settings.telegram_session_string:
        logger.warning("WEB_WATCH_DELETED set but TELEGRAM_SESSION_STRING is empty; skipping watcher.")
        return
    from .deleted_watcher import DeletedWatcherManager
    from .telegram_client import build_client_from_session_string

    def _build(user: dict) -> object:
        return build_client_from_session_string(
            settings, user["session_string"], api_id=user.get("api_id"), api_hash=user.get("api_hash"),
        )

    manager = DeletedWatcherManager(repository=state.repository, build_client=_build)
    manager.start_for_user({
        "bot_user_id": WEB_OWNER_ID,
        "session_string": settings.telegram_session_string,
        "api_id": settings.telegram_api_id,
        "api_hash": settings.telegram_api_hash,
    })
    logger.info("Started server-account deleted-message watcher (WEB_WATCH_DELETED).")


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    setup_logging()
    try:
        _maybe_start_web_watcher()
    except Exception:
        logger.exception("Could not start web deleted-message watcher")
    server = ThreadingHTTPServer((host, port), RequestHandler)
    logger.info("Serving on http://%s:%s", host, port)
    server.serve_forever()


if __name__ == "__main__":
    run()
