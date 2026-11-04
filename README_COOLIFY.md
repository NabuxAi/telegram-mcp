# استقرار روی Coolify

Coolify خودش HTTPS، دامنه و ریورس‌پروکسی را می‌دهد؛ Caddy لازم نیست. از `docker-compose.coolify.yml` استفاده کن.

## مرحله‌به‌مرحله
1. **ریپو را به Git بده** (خصوصی هم اشکال ندارد). فایل‌های لازم: `Dockerfile`, `server.py`, `app/`, `telegram_notebook/`, `requirements.txt`, `docker-compose.coolify.yml`.
2. در Coolify: **New Resource → Docker Compose**، ریپو را وصل کن و مسیر compose را `docker-compose.coolify.yml` بگذار.
3. **Environment Variables** را پر کن:
   ```
   TG_API_ID=<از my.telegram.org>
   TG_API_HASH=<از my.telegram.org>
   PUBLIC_BASE_URL=https://tg-mcp.yourdomain.com      # همان دامنه‌ای که در مرحلهٔ ۴ می‌دهی، بدون / آخر
   APP_SECRET_KEY=<خروجی: python3 -c "import secrets; print(secrets.token_urlsafe(48))">
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=<کلید>        # یا LLM_PROVIDER=openai + OPENAI_API_KEY
   ```
   > این مقادیر را فقط در Coolify بگذار؛ هیچ‌وقت داخل مخزن کامیت نکن. `APP_SECRET_KEY` را جایی امن نگه دار: با تغییرش همهٔ نشست‌ها باطل می‌شوند.
4. **دامنه و پورت:** به سرویس `mcp` دامنه بده و پورت را **8000** بگذار.
5. **Persistent Storage:** والیوم `tg-data` روی `/data` (در compose تعریف شده) — نشست‌ها، توکن‌ها و آرشیو نوت‌بوک آن‌جاست.
6. **Deploy**. health-check روی `/healthz` سبز می‌شود.
7. آدرس MCP برای کاربرها:
   ```
   https://tg-mcp.yourdomain.com/mcp
   ```
   هر کاربر در claude.ai → Settings → Connectors → Add custom connector همین را می‌دهد و با شمارهٔ خودش وارد می‌شود. دیگر لاگین دستی روی سرور لازم نیست.

## ربات تلگرام (اختیاری)
اگر می‌خواهی همان آرشیو از داخل تلگرام هم با ربات کار کند: `TELEGRAM_BOT_TOKEN` را اضافه کن و سرویس `bot` را (پروفایل `bot`) فعال کن. ورود کاربر از وب، ربات را هم برای او فعال می‌کند.

## بررسی سلامت
- `https://<دامنه>/healthz` → `ok`
- `https://<دامنه>/` → صفحهٔ راهنمای اتصال
- `https://<دامنه>/.well-known/oauth-authorization-server` → متادیتای OAuth (JSON)
- `POST /mcp` بدون توکن → 401 (طبیعی است)

## نکات
- اگر لاگین کاربر با «Authorization request expired» برمی‌گردد، `PUBLIC_BASE_URL` با دامنهٔ واقعی فرق دارد یا HTTPS برقرار نیست.
- برای بستن دسترسی یک کاربر: خودش از داشبورد «قطع اتصال» می‌زند، یا از تلگرام Settings → Devices نشست را Terminate می‌کند.
