> **Community source release (MIT).** Self-host using your own credentials. No NabuxAI production accounts or hosted-service access are included.

# Telegram MCP — تلگرامِ خودت + یک مغز NotebookLM، برای هر کاربری

یک سرور MCP چندکاربره: **هر کسی** با شمارهٔ تلگرامش وارد می‌شود و از آن به بعد Claude (وب، دسکتاپ، Claude Code)، Cursor یا هر کلاینت MCP دیگری از طرف **حساب خودِ او** با تلگرام کار می‌کند. کنارش یک «نوت‌بوک» شخصی هست که کانال‌ها و چت‌ها را ایندکس می‌کند و به سؤال‌ها با استناد جواب می‌دهد.

سرور و اپ ما فقط واسط هستند: نشست تلگرام هر کاربر **رمزنگاری‌شده روی همین سرور** می‌ماند، هیچ‌وقت به Claude داده نمی‌شود، و کاربر از داشبورد یا از تنظیمات تلگرام (Devices) هر وقت خواست قطعش می‌کند.

```
کلاینت MCP (Claude / Cursor / ...)
        │  OAuth 2.1 (PKCE, dynamic client registration)
        ▼
  این سرور  ──►  صفحهٔ لاگین: شماره → کد → رمز دومرحله‌ای
        │
        ├─ telegram_*  ابزارها  ──►  Telethon (نشست شخصی کاربر)  ──►  تلگرام
        └─ notebook_*  ابزارها  ──►  آرشیو per-user + جست‌وجوی معنایی + RAG با LLM
```

## چطور یک کاربر وصل می‌شود؟
1. در claude.ai → **Settings → Connectors → Add custom connector** و آدرس زیر را می‌دهد:
   ```
   https://<دامنه‌ات>/mcp
   ```
   (در Claude Code: `claude mcp add --transport http telegram https://<دامنه‌ات>/mcp`)
2. روی **Connect** می‌زند؛ مرورگر به صفحهٔ لاگین ما می‌آید. شماره، کدی که تلگرام فرستاده و (اگر داشته باشد) رمز دومرحله‌ای را وارد می‌کند.
3. تمام. Claude حالا با حساب همان شخص کار می‌کند. اگر بعداً کلاینت دومی وصل کند، فقط صفحهٔ «اجازه می‌دهم / رد» را می‌بیند.

کلاینتی که OAuth بلد نیست؟ کاربر بعد از ورود از **داشبورد** (`/dashboard`) یک توکن شخصی می‌سازد و یکی از این دو را استفاده می‌کند:
- `Authorization: Bearer <token>` روی `https://<دامنه>/mcp`
- یا بدون هدر: `https://<دامنه>/t/<token>/mcp`

## تاپیک‌های گروه‌های Forum (انتشار در بخش مشخص)

برای دیدن بخش‌های واقعی یک سوپرگروه که Topics فعال دارد، **بدون خواندن پیام‌ها**، ابزار
`telegram_list_forum_topics` را با `chat="@group"` و حساب درست (`account="@your_account"`) فراخوانی کنید.
این ابزار `id`، `title`، وضعیت بسته/مخفی بودن و در صورت امکان `next_offset` را برمی‌گرداند.
برای ادامهٔ صفحه‌بندی، سه مقدار `offset_date`، `offset_id` و `offset_topic` را از
`next_offset` به درخواست بعدی بدهید.

برای انتشار متن یا عکس/فایل در همان بخش، `telegram_send_message` را با
`chat="@group"` و `topic_id=<id from listing>` فراخوانی کنید.
تاپیک `General` با `topic_id=1` در خط اصلی گروه ارسال می‌شود؛ سایر تاپیک‌ها با پاسخ
به پیام ریشهٔ تاپیک مقصد مشخص می‌شوند.

**پیش از ارسال:** حساب، گروه و نام تاپیک را با کاربر تأیید کنید. سرور پیش از ارسال
وجود تاپیک را در همان گروه بررسی می‌کند و تاپیک ناموجود/حذف‌شده، بسته یا مخفی
را رد می‌کند. `topic_id` و `reply_to` را هم‌زمان نفرستید. بدون `topic_id`
رفتار قدیمی ارسال معمولی/ریپلای حفظ می‌شود. برای گروه‌های غیر Forum،
درخواست فهرست تاپیک یا ارسال با `topic_id` رد می‌شود.

## مدیریت کامل تاپیک‌ها (Telethon / Forum API)

این نسخه علاوه بر فهرست‌کردن تاپیک‌ها، ابزارهای مدیریتی زیر را با حساب
متصلِ همان کاربر و **بررسی حق Manage Topics** ارائه می‌دهد:

| ابزار | کاربرد |
|---|---|
| `telegram_list_forum_topics` | فهرست واقعی تاپیک‌ها، وضعیت و صفحه‌بندی |
| `telegram_create_forum_topic` | ساخت تاپیک جدید، همراه جلوگیری از عنوان تکراری |
| `telegram_edit_forum_topic` | تغییر عنوان/ایموجی، بستن و بازکردن، مخفی‌سازی General |
| `telegram_pin_forum_topic` | پین یا آن‌پین تاپیک |
| `telegram_delete_forum_topic` | **حذف برگشت‌ناپذیر تاپیک و همه پیام‌ها** با `confirm_delete=true` |
| `telegram_set_forum_enabled` | تغییر حالت Forum برای **مالک سوپرگروه** با `confirm_change=true` |
| `telegram_get_messages(topic_id=...)` | مطالعه پیام‌های تاپیک خاص |
| `telegram_send_message(topic_id=...)` | ارسال متن، عکس، فایل یا پیام زمان‌بندی‌شده به تاپیک خاص |
| `telegram_send_poll(topic_id=...)` | ارسال نظرسنجی یا آزمون به تاپیک خاص |

ابتدا `telegram_list_forum_topics` را برای پیدا کردن `topic_id` اجرا کنید
و از حدس‌زدن شماره‌ها خودداری کنید. برای جلوگیری از تکرار، ابزار ساخت عنوان
دقیقاً یکسان را رد می‌کند. هیچ عملیاتی روی حسابی خارج از حساب‌های متصلِ مالک
توکن OAuth انجام نمی‌شود. تست‌ها آفلاین هستند و پیام واقعی به تلگرام نمی‌فرستند.

**نکتهٔ استقرار:** اگر متدهای جدید را در ChatGPT نمی‌بینید، نسخه جدید سرور را
deploy کنید و سپس اتصال Plugin/MCP را Refresh یا Reconnect کنید تا فهرست ابزارهای
سرور دوباره شناسایی شود؛ تغییر GitHub به تنهایی نسخهٔ در حال اجرا را به‌روز نمی‌کند.

## ابزارها

### `telegram_*` — عملیات روی تلگرامِ خود کاربر
| ابزار | کار |
|---|---|
| `telegram_me` | حساب متصل |
| `telegram_list_chats` | فهرست چت‌ها (فیلتر نوع، جست‌وجوی عنوان، فقط خوانده‌نشده، آرشیو) |
| `telegram_get_chat` | اطلاعات کامل کاربر/گروه/کانال |
| `telegram_search_contacts` / `telegram_get_contacts` / `telegram_add_contact` / `telegram_block_user` | مخاطبین |
| `telegram_get_messages` | تاریخچهٔ یک چت (صفحه‌بندی، جست‌وجو، فرستندهٔ خاص، idهای مشخص، ریپلای‌ها) |
| `telegram_search_messages` | جست‌وجوی متن در یک چت یا سراسر حساب |
| `telegram_send_message` | ارسال متن/فایل/عکس، ریپلای، بی‌صدا، **زمان‌بندی‌شده** (`schedule_at`) |
| `telegram_send_poll` | نظرسنجی / کوییز |
| `telegram_edit_message` | ویرایش متن و/یا جایگزینی عکس (پست کانال هم) |
| `telegram_delete_messages` / `telegram_forward_messages` / `telegram_react` / `telegram_mark_read` / `telegram_pin_message` | مدیریت پیام |
| `telegram_download_media` | دریافت عکس (به‌صورت تصویر برای مدل)، فایل متنی (به‌صورت متن)، یا متادیتا |
| `telegram_get_participants` | اعضای گروه/کانال (all / admins / bots / kicked / banned) |
| `telegram_create_chat` / `telegram_invite_users` / `telegram_join_chat` / `telegram_leave_chat` | ساخت و مدیریت گروه و کانال |
| `telegram_archive_chat` / `telegram_mute_chat` | آرشیو و بی‌صدا کردن |
| `telegram_update_profile` | نام، بیو، یوزرنیم |
| `telegram_backup_messages` / `telegram_restore_message` | بکاپ متن+عکس اصلی روی سرور قبل از ویرایش انبوه، و بازگردانی |

`chat` را می‌توان به شکل `@username`، آیدی عددی (خروجی `telegram_list_chats`)، لینک `t.me`، شمارهٔ یک مخاطب، یا `me` (Saved Messages) داد.

### `notebook_*` — مغز (از پروژهٔ Telegram NotebookLM)
| ابزار | کار |
|---|---|
| `notebook_import_chat` | ایندکس یک کانال/گروه/چت در آرشیو شخصی: متن پیام‌ها + رونویسی صوت/ویدیو + OCR عکس/PDF (با کلید Gemini) + DOCX/XLSX. `wait=true` برای importهای کوچک، وگرنه جاب پس‌زمینهٔ قابل ادامه |
| `notebook_import_jobs` / `notebook_cancel_import` | وضعیت/لغو جاب‌ها |
| `notebook_index_messages` | ایندکس چند پیام مشخص از هر چتی («اینو یادت باشه») |
| `notebook_save_note` | ذخیرهٔ یادداشت آزاد در inbox نوت‌بوک |
| `notebook_search` | جست‌وجوی معنایی + کلیدواژه‌ای روی همهٔ محتوای ایندکس‌شده |
| `notebook_ask` | پاسخ مستند با ارجاع `[n]` از روی آرشیو (LLM سرور) |
| `notebook_summarize` / `notebook_topics` / `notebook_timeline` / `notebook_stats` / `notebook_recent` / `notebook_get_item` / `notebook_export` | خلاصه، خوشه‌بندی موضوعی، تایم‌لاین، آمار، خروجی Markdown |
| `notebook_sources` / `notebook_delete_source` | منابع ایندکس‌شده |
| `notebook_tags` / `notebook_rules` / `notebook_add_rule` / `notebook_remove_rule` | تگ‌گذاری خودکار (کلیدواژه → تگ) |

آرشیو با **آیدی تلگرام کاربر** کلید می‌خورد؛ همان کلیدی که ربات تلگرامِ پروژهٔ نوت‌بوک استفاده می‌کند. پس کسی که از وب وصل شده و کسی که با ربات (`/connect`) کار می‌کند یک آرشیو مشترک دارند، و ورود از وب خودکار ربات را هم برای همان کاربر فعال می‌کند.

## پیش‌نیاز
- یک سرور با **Docker** (یا Python 3.11+ و ffmpeg).
- یک دامنهٔ HTTPS (کانکتور Claude و OAuth به HTTPS نیاز دارند). با Caddy داخل compose خودکار گرفته می‌شود؛ یا Coolify؛ یا Cloudflare Tunnel.
- `api_id` / `api_hash` از [my.telegram.org](https://my.telegram.org) — **یک جفت برای کل سرور**؛ کاربرها فقط شماره‌شان را می‌دهند (در بخش «پیشرفته» صفحهٔ لاگین می‌توانند اعتبار خودشان را هم بدهند).
- برای «مغز»: یک ارائه‌دهندهٔ LLM/embedding (Gemini یا OpenAI با کلید، یا Ollama محلی).

## راه‌اندازی با Docker
```bash
cp .env.example .env && nano .env
```
حداقل این‌ها را پر کن:
```
TG_API_ID=...            TG_API_HASH=...
PUBLIC_BASE_URL=https://mcp.example.com
APP_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(48))")
LLM_PROVIDER=gemini      GEMINI_API_KEY=...      # یا openai / ollama
```
در `Caddyfile` دامنه‌ات را بگذار، بعد:
```bash
docker compose up -d --build
```
آدرس MCP: `https://mcp.example.com/mcp` — همین را به کاربرها بده. صفحهٔ `https://mcp.example.com/` راهنمای اتصال را نشان می‌دهد.

ربات تلگرام (اختیاری، برای همان آرشیو): `TELEGRAM_BOT_TOKEN` را در `.env` بگذار و
```bash
docker compose --profile bot up -d
```

> کاربر Coolify؟ راهنمای `README_COOLIFY.md` و فایل `docker-compose.coolify.yml`.

## بدون Docker
```bash
cp .env.example .env   # DATA_DIR=./data ؛ برای تست محلی PUBLIC_BASE_URL=http://localhost:8000 مجاز است
pip install -r requirements.txt
./run_local.sh
```

## امنیت — چیزهایی که باید بدانی
- **`APP_SECRET_KEY`** نشست‌های تلگرام را (Fernet) رمز می‌کند و کوکی‌ها را امضا می‌کند. عوضش کنی همه باید دوباره وارد شوند. گمش کنی داده‌ها غیرقابل‌خواندن‌اند.
- توکن‌های OAuth فقط به‌صورت hash ذخیره می‌شوند؛ access token یک ساعت، refresh token ۹۰ روز و با هر refresh عوض می‌شود. لغو از داشبورد یا `/revoke`.
- هر کاربر فقط به نشست خودش می‌رسد: user_id از توکن می‌آید، نه از ورودی ابزار.
- آدرس‌های `file_url` را خودِ سرور دانلود می‌کند؛ محافظ SSRF (فقط IP عمومی، ≤۳ ریدایرکت، سقف حجم) روشن است.
- محدودیت درخواست کد (`LOGIN_CODES_PER_HOUR` به‌ازای IP) و لیست سفید شماره (`ALLOWED_PHONES`) اختیاری.
- همیشه پشت HTTPS. `PUBLIC_BASE_URL` با `http://` فقط برای `localhost` پذیرفته می‌شود.

## تست
```bash
python -m pytest -q test_server.py     # کل مسیر OAuth + لاگین + ابزارها، آفلاین با Telethon ساختگی
python -m pytest -q tests/notebook     # تست‌های موتور نوت‌بوک
```

## ساختار
```
server.py              entrypoint (FastMCP + OAuth + uvicorn، مسیر /t/<token>/mcp)
app/config.py          تنظیمات از env
app/db.py              SQLite: کاربران (نشست رمزشده)، کلاینت‌های OAuth، کدها، توکن‌ها
app/security.py        Fernet، hash توکن، کوکی امضاشده، CSRF
app/oauth.py           OAuthAuthorizationServerProvider روی SQLite + تراکنش‌های /authorize
app/tg.py              استخر کلاینت Telethon per-user، جریان لاگین، resolve/serialize، fetch امن
app/tools.py           ابزارهای telegram_*
app/notebook.py        پل به موتور نوت‌بوک + ابزارهای notebook_*
app/web.py             صفحات لاگین / consent / داشبورد
telegram_notebook/     موتور NotebookLM (ingest، search، RAG، bot، …) — وارد‌شده از پروژهٔ telegram-notebooklm-mvp
```
