"""Browser-facing pages: Telegram login (phone -> code -> 2FA), OAuth consent,
and a small per-user dashboard (personal token, connected apps, disconnect).

All pages are plain HTML forms — no JavaScript needed — so they work inside the
embedded browser MCP clients open for OAuth.
"""
import contextlib
import html
import logging
from typing import Optional

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response

from . import config, security
from .db import get_db
from .oauth import PERSONAL_CLIENT_ID, TelegramOAuthProvider
from .tg import ClientPool, LoginError, LoginManager

log = logging.getLogger("telegram_mcp.web")
COOKIE = "tgmcp_session"


def e(s) -> str:
    return html.escape(str(s if s is not None else ""), quote=True)


_CSS = """
:root{--bg:#0f1419;--card:#1a2129;--fg:#e8edf2;--muted:#8b98a5;--acc:#2aabee;--err:#ff6b6b;--ok:#3ddc97;--bd:#2c3640}
@media (prefers-color-scheme: light){:root{--bg:#f4f6f8;--card:#fff;--fg:#14202b;--muted:#5b6b7a;--bd:#dbe2e8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 -apple-system,Segoe UI,Roboto,Vazirmatn,Tahoma,sans-serif}
main{max-width:520px;margin:6vh auto;padding:0 16px}.card{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:28px}
h1{font-size:22px;margin:0 0 6px}h2{font-size:17px;margin:24px 0 8px}p{margin:8px 0;color:var(--muted)}p.fg{color:var(--fg)}
label{display:block;margin:14px 0 6px;font-weight:600}input[type=text],input[type=tel],input[type=password]{width:100%;padding:12px;border-radius:10px;border:1px solid var(--bd);background:var(--bg);color:var(--fg);font-size:18px;direction:ltr;text-align:left}
button,.btn{display:inline-block;margin-top:16px;padding:12px 20px;border:0;border-radius:10px;background:var(--acc);color:#fff;font-size:16px;font-weight:600;cursor:pointer;text-decoration:none}
button.sec,.btn.sec{background:transparent;border:1px solid var(--bd);color:var(--fg)}button.danger{background:var(--err)}
.err{background:rgba(255,107,107,.12);border:1px solid var(--err);color:var(--err);padding:10px 12px;border-radius:10px;margin:12px 0}
.ok{background:rgba(61,220,151,.12);border:1px solid var(--ok);padding:10px 12px;border-radius:10px;margin:12px 0}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;direction:ltr;display:block;background:var(--bg);border:1px solid var(--bd);padding:10px;border-radius:10px;word-break:break-all;font-size:14px;text-align:left}
details{margin-top:14px}summary{cursor:pointer;color:var(--muted)}table{width:100%;border-collapse:collapse;margin-top:8px}td,th{padding:8px 6px;border-bottom:1px solid var(--bd);text-align:right;font-size:14px;vertical-align:top}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.small{font-size:13px}.logo{width:44px;height:44px;border-radius:12px;background:var(--acc);display:inline-flex;align-items:center;justify-content:center;margin-bottom:12px}
"""


def page(title: str, body: str, *, lang: str = "fa") -> HTMLResponse:
    rtl = ' dir="rtl"' if lang == "fa" else ""
    doc = (
        f'<!doctype html><html lang="{lang}"{rtl}><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{e(title)} · {e(config.APP_NAME)}</title><style>{_CSS}</style></head>"
        f'<body><main><div class="card"><div class="logo">✈️</div>{body}</div>'
        f'<p class="small" style="text-align:center;margin-top:14px">{e(config.APP_NAME)}</p></main></body></html>'
    )
    headers = {"Cache-Control": "no-store", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer"}
    return HTMLResponse(doc, headers=headers)


def err_box(msg: Optional[str]) -> str:
    return f'<div class="err">{e(msg)}</div>' if msg else ""


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "?"


def set_cookie(resp: Response, user_id: int) -> None:
    resp.set_cookie(COOKIE, security.sign_cookie(user_id), max_age=config.COOKIE_TTL, httponly=True,
                    secure=config.PUBLIC_BASE_URL.startswith("https://"), samesite="lax", path="/")


def clear_cookie(resp: Response) -> None:
    resp.delete_cookie(COOKIE, path="/")


def current_user(request: Request) -> Optional[dict]:
    uid = security.verify_cookie(request.cookies.get(COOKIE))
    if uid is None:
        return None
    u = get_db().get_user(uid)
    return u


def csrf_of(request: Request) -> str:
    return security.csrf_token(request.cookies.get(COOKIE, ""))


def csrf_ok(request: Request, form) -> bool:
    return security.check_csrf(request.cookies.get(COOKIE), form.get("csrf"))


# ----------------------------------------------------------------------------
# Page bodies
# ----------------------------------------------------------------------------
def landing_body() -> str:
    return (
        f"<h1>اتصال تلگرام به هوش مصنوعی</h1>"
        f"<p>با ورود به حساب تلگرامت، ابزارهای MCP این سرور به نمایندگی از <b>خودت</b> با تلگرام کار می‌کنند: "
        f"خواندن و ارسال پیام، مدیریت گروه و کانال، جست‌وجو، فایل‌ها و بیشتر.</p>"
        f"<p>نشست تلگرامت رمزنگاری‌شده روی همین سرور می‌ماند و هر زمان می‌توانی از داشبورد یا از تنظیمات تلگرام (Devices) قطعش کنی.</p>"
        f'<a class="btn" href="/login">ورود با شمارهٔ تلگرام</a>'
        f'<h2>چطور وصل شوم؟</h2>'
        f'<p class="fg">۱) در Claude (وب، دسکتاپ یا Claude Code) یک <b>Custom Connector</b> با این آدرس بساز:</p>'
        f"<code>{e(config.PUBLIC_BASE_URL)}{config.MCP_PATH}</code>"
        f'<p class="fg">۲) روی Connect بزن؛ به همین‌جا می‌آیی، شماره و کد تلگرام را وارد می‌کنی، تمام.</p>'
        f'<p class="small">کلاینتی داری که OAuth بلد نیست؟ بعد از ورود، از داشبورد یک توکن شخصی بگیر.</p>'
    )


def phone_form(txn: Optional[str], error: Optional[str] = None, phone: str = "") -> str:
    adv = ""
    if config.ALLOW_USER_API_CREDENTIALS:
        adv = (
            "<details><summary>پیشرفته: استفاده از api_id/api_hash خودم</summary>"
            "<p class='small'>اختیاری. اگر خالی بگذاری از اعتبار خود سرور استفاده می‌شود. از my.telegram.org بگیر.</p>"
            '<label>api_id</label><input type="text" name="api_id" inputmode="numeric" autocomplete="off">'
            '<label>api_hash</label><input type="text" name="api_hash" autocomplete="off"></details>'
        )
    return (
        "<h1>ورود به تلگرام</h1>"
        "<p>شماره‌ات را با کد کشور وارد کن. تلگرام یک کد تأیید برایت می‌فرستد (داخل اپ تلگرام یا پیامک).</p>"
        + err_box(error) +
        '<form method="post" action="/login/phone">'
        f'<input type="hidden" name="txn" value="{e(txn or "")}">'
        '<label>شمارهٔ تلفن</label>'
        f'<input type="tel" name="phone" value="{e(phone)}" placeholder="+989121234567" autofocus required autocomplete="tel">'
        + adv +
        '<button type="submit">ارسال کد</button></form>'
    )


def code_form(flow_id: str, phone: str, error: Optional[str] = None) -> str:
    return (
        "<h1>کد تأیید</h1>"
        f"<p>کدی که تلگرام به <b dir='ltr'>{e(phone)}</b> فرستاد را وارد کن.</p>"
        + err_box(error) +
        '<form method="post" action="/login/code">'
        f'<input type="hidden" name="flow" value="{e(flow_id)}">'
        '<label>کد</label><input type="text" name="code" inputmode="numeric" autocomplete="one-time-code" autofocus required>'
        '<button type="submit">تأیید</button></form>'
        '<form method="post" action="/login/cancel" style="display:inline">'
        f'<input type="hidden" name="flow" value="{e(flow_id)}"><button class="sec" type="submit">شمارهٔ دیگر</button></form>'
    )


def password_form(flow_id: str, error: Optional[str] = None) -> str:
    return (
        "<h1>رمز دومرحله‌ای</h1>"
        "<p>این حساب تأیید دومرحله‌ای دارد. رمز عبور ابری تلگرامت را وارد کن.</p>"
        + err_box(error) +
        '<form method="post" action="/login/password">'
        f'<input type="hidden" name="flow" value="{e(flow_id)}">'
        '<label>رمز</label><input type="password" name="password" autofocus required autocomplete="current-password">'
        '<button type="submit">ورود</button></form>'
    )


def consent_body(user: dict, txn, csrf: str) -> str:
    name = " ".join(x for x in [user.get("first_name"), user.get("last_name")] if x) or user.get("phone")
    return (
        f"<h1>اجازهٔ دسترسی</h1>"
        f"<p><b>{e(txn.client_name)}</b> می‌خواهد از طرف حساب تلگرام <b>{e(name)}</b> "
        f"(<span dir='ltr'>{e(user.get('phone'))}</span>) با تلگرام کار کند.</p>"
        f'<form method="post" action="/login/consent"><input type="hidden" name="txn" value="{e(txn.id)}">'
        f'<input type="hidden" name="csrf" value="{e(csrf)}"><div class="row">'
        f'<button type="submit" name="decision" value="allow">اجازه می‌دهم</button>'
        f'<button type="submit" name="decision" value="deny" class="sec">رد</button></div></form>'
        f'<p class="small"><a href="/login?txn={e(txn.id)}&fresh=1">ورود با حساب دیگر</a></p>'
    )


def _accounts_block(accounts: list[dict], current_id: int, csrf: str) -> str:
    """The person's connected Telegram accounts, in connection order.

    Position is what the MCP tools and the API use to address an account, so it
    is shown here rather than left implicit — someone reading "2" in an API
    response needs to be able to see which phone that is.
    """
    rows = ""
    for i, a in enumerate(accounts, start=1):
        nm = " ".join(x for x in (a.get("first_name"), a.get("last_name")) if x) or "—"
        un = f" (@{e(a['username'])})" if a.get("username") else ""
        badge = " <span class='small'>· این حساب</span>" if a["id"] == current_id else ""
        dead = "" if a.get("session_ok") else " <span class='err small'>نشست باطل</span>"
        switch = ""
        if a["id"] != current_id:
            switch = (f"<form method='post' action='/dashboard/switch'><input type='hidden' name='csrf' value='{e(csrf)}'>"
                      f"<input type='hidden' name='id' value='{a['id']}'><button class='sec small' type='submit'>باز کردن</button></form>")
        rows += (f"<tr><td><b>{i}</b></td><td>{e(nm)}{un}{badge}{dead}<br>"
                 f"<span class='small' dir='ltr'>{e(a.get('phone') or '')}</span></td><td>{switch}</td></tr>")
    return (
        "<h2>حساب‌های تلگرام</h2>"
        "<p class='small'>می‌توانی چند حساب وصل کنی. شمارهٔ ردیف همان چیزی است که در ابزارها و API با "
        "<span dir='ltr'>account</span> صدا می‌زنی؛ بدون آن، حساب ردیف ۱ استفاده می‌شود.</p>"
        f"<table><tr><th>#</th><th>حساب</th><th></th></tr>{rows}</table>"
        "<p><a class='btn sec' href='/login?add=1'>وصل کردن حساب دیگر</a></p>"
    )


def dashboard_body(user: dict, tokens: list[dict], csrf: str, new_token: Optional[str] = None,
                   notice: Optional[str] = None, stats: Optional[dict] = None,
                   accounts: Optional[list[dict]] = None) -> str:
    name = " ".join(x for x in [user.get("first_name"), user.get("last_name")] if x) or "—"
    uname = f" (@{e(user['username'])})" if user.get("username") else ""
    mcp_url = f"{config.PUBLIC_BASE_URL}{config.MCP_PATH}"
    rows = ""
    for t in tokens:
        if t["kind"] != "access":
            continue
        label = t["label"] or t["client_name"]
        kind = "توکن شخصی" if t["client_id"] == PERSONAL_CLIENT_ID else "OAuth"
        rows += (
            f"<tr><td>{e(label)}<br><span class='small'>{kind}</span></td>"
            f"<td class='small'>{_ago(t.get('last_used_at'))}</td>"
            f"<td><form method='post' action='/dashboard/revoke'><input type='hidden' name='csrf' value='{e(csrf)}'>"
            f"<input type='hidden' name='h' value='{e(t['token_hash'])}'><button class='sec small' type='submit'>لغو</button></form></td></tr>"
        )
    table = f"<table><tr><th>اپ / توکن</th><th>آخرین استفاده</th><th></th></tr>{rows}</table>" if rows else "<p>هنوز هیچ اپی وصل نشده.</p>"
    warn = "" if user.get("session_ok") else '<div class="err">نشست تلگرام این حساب باطل شده. دوباره <a href="/login?fresh=1">وارد شو</a>.</div>'
    tokbox = ""
    if new_token:
        tokbox = (
            '<div class="ok">توکن ساخته شد. فقط همین یک‌بار نشان داده می‌شود؛ کپی‌اش کن.</div>'
            f"<code>{e(new_token)}</code>"
            "<p class='small'>روش ۱ (هدر): <span dir='ltr'>Authorization: Bearer &lt;token&gt;</span> روی آدرس بالا.</p>"
            f"<p class='small'>روش ۲ (بدون هدر):</p><code>{e(config.PUBLIC_BASE_URL)}/t/{e(new_token)}{config.MCP_PATH}</code>"
        )
    return (
        f"<h1>داشبورد</h1>{warn}"
        + (f'<div class="ok">{e(notice)}</div>' if notice else "")
        + f"<p class='fg'><b>{e(name)}</b>{uname} · <span dir='ltr'>{e(user.get('phone'))}</span></p>"
        f"<h2>آدرس MCP</h2><code>{e(mcp_url)}</code>"
        "<p class='small'>در Claude → Settings → Connectors → Add custom connector همین آدرس را بده و Connect را بزن.</p>"
        + _accounts_block(accounts or [user], user["id"], csrf)
        + _stats_block(stats) +
        f"<h2>اپ‌های متصل</h2>{table}"
        "<h2>توکن شخصی</h2><p>برای کلاینت‌هایی که OAuth ندارند (مثلاً اسکریپت خودت).</p>"
        + tokbox +
        f"<form method='post' action='/dashboard/token'><input type='hidden' name='csrf' value='{e(csrf)}'>"
        "<label>نام (اختیاری)</label><input type='text' name='label' placeholder='مثلاً: لپ‌تاپ کار'>"
        "<button type='submit'>ساخت توکن</button></form>"
        "<h2>قطع اتصال</h2><p>نشست تلگرام از این سرور پاک و در تلگرام هم خاتمه داده می‌شود؛ همهٔ توکن‌ها باطل می‌شوند.</p>"
        "<div class='row'>"
        f"<form method='post' action='/dashboard/disconnect' onsubmit='return confirm(\"مطمئنی؟\")'><input type='hidden' name='csrf' value='{e(csrf)}'><button class='danger' type='submit'>قطع اتصال تلگرام</button></form>"
        f"<form method='post' action='/logout'><input type='hidden' name='csrf' value='{e(csrf)}'><button class='sec' type='submit'>خروج از داشبورد</button></form>"
        "</div>"
    )


def _stats_block(stats: Optional[dict]) -> str:
    if not stats:
        return ""
    items = stats.get("items") or stats.get("total_items") or 0
    sources = stats.get("sources") or stats.get("source_count") or 0
    tags = stats.get("tags") or stats.get("tag_count") or 0
    if isinstance(sources, list):
        sources = len(sources)
    if isinstance(tags, list):
        tags = len(tags)
    return (f"<h2>نوت‌بوک (مغز)</h2><p class='fg'>{e(items)} آیتم ایندکس‌شده از {e(sources)} منبع · {e(tags)} تگ</p>"
            "<p class='small'>از داخل Claude بگو: «کانال X را وارد نوت‌بوک کن» یا «از نوت‌بوکم بپرس …».</p>")


def _ago(ts: Optional[int]) -> str:
    if not ts:
        return "—"
    import time
    d = int(time.time()) - int(ts)
    if d < 60:
        return "همین الان"
    if d < 3600:
        return f"{d // 60} دقیقه پیش"
    if d < 86400:
        return f"{d // 3600} ساعت پیش"
    return f"{d // 86400} روز پیش"


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
def register(mcp, provider: TelegramOAuthProvider, logins: LoginManager, pool: ClientPool,
             on_disconnect=None, notebook_stats=None) -> None:
    db = get_db

    async def finish_login(request: Request, user_id: int, txn_id: Optional[str]) -> Response:
        if txn_id:
            try:
                url = provider.complete_transaction(txn_id, user_id)
            except ValueError as ex:
                resp = page("خطا", f"<h1>خطا</h1>{err_box(str(ex))}<a class='btn' href='/dashboard'>داشبورد</a>")
                set_cookie(resp, user_id)
                return resp
            resp = RedirectResponse(url, status_code=302)
        else:
            resp = RedirectResponse("/dashboard", status_code=302)
        set_cookie(resp, user_id)
        return resp

    @mcp.custom_route("/", methods=["GET"])
    async def root(request: Request):
        if current_user(request):
            return RedirectResponse("/dashboard", status_code=302)
        return page("اتصال تلگرام", landing_body())

    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(request: Request):
        return PlainTextResponse("ok")

    @mcp.custom_route("/login", methods=["GET"])
    async def login_get(request: Request):
        txn_id = request.query_params.get("txn") or None
        # `add=1` is "connect another account": show the phone form even though
        # a session exists, and keep that session so the new account joins this
        # person's set instead of starting a separate identity. `fresh=1` is the
        # older "log in as someone else" and still means exactly that.
        add = request.query_params.get("add") == "1"
        fresh = request.query_params.get("fresh") == "1" or add
        txn = provider.get_transaction(txn_id) if txn_id else None
        if txn_id and txn is None:
            return page("خطا", "<h1>درخواست منقضی شد</h1><p>دوباره از داخل کلاینت MCP روی Connect بزن.</p>")
        user = current_user(request)
        if user and user.get("session_ok") and not fresh:
            if txn:
                return page("اجازهٔ دسترسی", consent_body(user, txn, csrf_of(request)))
            return RedirectResponse("/dashboard", status_code=302)
        hint = ("<p class='small'>این شماره به حساب‌های فعلی‌ات اضافه می‌شود.</p>"
                if (add and user) else "")
        return page("ورود", hint + phone_form(txn_id))

    @mcp.custom_route("/login/phone", methods=["POST"])
    async def login_phone(request: Request):
        form = await request.form()
        txn_id = (form.get("txn") or "").strip() or None
        phone = (form.get("phone") or "").strip()
        api_id = api_hash = None
        if config.ALLOW_USER_API_CREDENTIALS:
            raw_id = (form.get("api_id") or "").strip()
            raw_hash = (form.get("api_hash") or "").strip()
            if raw_id or raw_hash:
                if not (raw_id.isdigit() and len(raw_hash) >= 16):
                    return page("ورود", phone_form(txn_id, "api_id باید عدد و api_hash معتبر باشد.", phone))
                api_id, api_hash = int(raw_id), raw_hash
        if txn_id and provider.get_transaction(txn_id) is None:
            return page("خطا", "<h1>درخواست منقضی شد</h1><p>دوباره از داخل کلاینت MCP روی Connect بزن.</p>")
        if logins.rate_limited(client_ip(request)):
            return page("ورود", phone_form(txn_id, "درخواست‌های زیادی از این آدرس ارسال شده. کمی بعد دوباره تلاش کن.", phone))
        # Already signed in? Then this is "add another account": the new phone
        # joins the caller's existing set rather than starting a new identity.
        # The owner comes from the session cookie only — never from the form —
        # so a crafted request cannot attach a phone to somebody else.
        me = current_user(request)
        owner_key = db().owner_of(me["id"]) if me else None
        try:
            flow = await logins.start(phone, txn=txn_id, api_id=api_id, api_hash=api_hash, owner_key=owner_key)
        except LoginError as ex:
            return page("ورود", phone_form(txn_id, str(ex), phone))
        return page("کد تأیید", code_form(flow.id, flow.phone))

    @mcp.custom_route("/login/code", methods=["POST"])
    async def login_code(request: Request):
        form = await request.form()
        flow_id = form.get("flow") or ""
        try:
            flow = logins.get(flow_id)
            status, user_id = await logins.submit_code(flow_id, form.get("code") or "")
        except LoginError as ex:
            if flow_id in logins.flows:
                f = logins.flows[flow_id]
                return page("کد تأیید", code_form(f.id, f.phone, str(ex)))
            return page("ورود", phone_form(None, str(ex)))
        if status == "password":
            return page("رمز دومرحله‌ای", password_form(flow.id))
        return await finish_login(request, user_id, flow.txn)

    @mcp.custom_route("/login/password", methods=["POST"])
    async def login_password(request: Request):
        form = await request.form()
        flow_id = form.get("flow") or ""
        try:
            flow = logins.get(flow_id)
            user_id = await logins.submit_password(flow_id, form.get("password") or "")
        except LoginError as ex:
            if flow_id in logins.flows:
                return page("رمز دومرحله‌ای", password_form(flow_id, str(ex)))
            return page("ورود", phone_form(None, str(ex)))
        return await finish_login(request, user_id, flow.txn)

    @mcp.custom_route("/login/cancel", methods=["POST"])
    async def login_cancel(request: Request):
        form = await request.form()
        flow_id = form.get("flow") or ""
        txn = None
        if flow_id in logins.flows:
            txn = logins.flows[flow_id].txn
        await logins.cancel(flow_id)
        return RedirectResponse("/login?fresh=1" + (f"&txn={txn}" if txn else ""), status_code=303)

    @mcp.custom_route("/login/consent", methods=["POST"])
    async def login_consent(request: Request):
        form = await request.form()
        user = current_user(request)
        if not user or not csrf_ok(request, form):
            return RedirectResponse("/login", status_code=302)
        txn_id = form.get("txn") or ""
        if form.get("decision") == "deny":
            url = provider.deny_transaction(txn_id)
            return RedirectResponse(url or "/dashboard", status_code=302)
        return await finish_login(request, user["id"], txn_id)

    @mcp.custom_route("/dashboard", methods=["GET"])
    async def dashboard(request: Request):
        user = current_user(request)
        if not user:
            return RedirectResponse("/login", status_code=302)
        owner = db().owner_of(user["id"])
        accounts = db().accounts_for_owner(owner) if owner else [user]
        return page("داشبورد", dashboard_body(user, db().list_tokens(user["id"]), csrf_of(request),
                                               stats=_nb_stats(user), accounts=accounts))

    @mcp.custom_route("/dashboard/switch", methods=["POST"])
    async def dashboard_switch(request: Request):
        """Look at the dashboard as one of your other accounts.

        Only ever moves between accounts that share an owner_key with the
        current session, so a posted id cannot select somebody else's account.
        """
        user = current_user(request)
        if not user:
            return RedirectResponse("/login", status_code=302)
        form = await request.form()
        if (form.get("csrf") or "") != csrf_of(request):
            return page("خطا", "<h1>درخواست نامعتبر</h1>")
        owner = db().owner_of(user["id"])
        target = str(form.get("id") or "")
        allowed = {str(a["id"]) for a in db().accounts_for_owner(owner)} if owner else set()
        if target not in allowed:
            return RedirectResponse("/dashboard", status_code=302)
        resp = RedirectResponse("/dashboard", status_code=302)
        set_cookie(resp, int(target))
        return resp

    @mcp.custom_route("/api/accounts", methods=["GET"])
    async def api_accounts(request: Request):
        """The caller's connected accounts, in the order they connected them.

        Same order and same identifiers the MCP tools use, so a client can list
        here and then pass `account` to a tool. Bearer token or dashboard
        cookie; the token's own account decides whose list this is.
        """
        user = _api_caller(request)
        if user is None:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        owner = db().owner_of(user["id"])
        rows = db().accounts_for_owner(owner) if owner else [user]
        return JSONResponse({"accounts": [
            {"position": i, "id": a["id"], "tg_user_id": a["tg_user_id"],
             "username": a.get("username"), "phone": a.get("phone"),
             "name": " ".join(x for x in (a.get("first_name"), a.get("last_name")) if x),
             "connected": bool(a.get("session_ok")),
             "current": a["id"] == user["id"]}
            for i, a in enumerate(rows, start=1)
        ], "default": 1 if rows else None})

    def _api_caller(request: Request) -> Optional[dict]:
        """Resolve an API caller from a Bearer token, falling back to the cookie."""
        auth = request.headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            d = db().get_token(auth[7:].strip(), kind="access")
            if d:
                return db().get_user(d["user_id"])
            return None
        return current_user(request)

    def _nb_stats(user):
        if notebook_stats is None:
            return None
        with contextlib.suppress(Exception):
            return notebook_stats(user["tg_user_id"])
        return None

    @mcp.custom_route("/dashboard/token", methods=["POST"])
    async def dashboard_token(request: Request):
        form = await request.form()
        user = current_user(request)
        if not user or not csrf_ok(request, form):
            return RedirectResponse("/login", status_code=302)
        label = (form.get("label") or "").strip()[:60] or "personal"
        token = provider.issue_personal_token(user["id"], label)
        return page("داشبورد", dashboard_body(user, db().list_tokens(user["id"]), csrf_of(request), new_token=token))

    @mcp.custom_route("/dashboard/revoke", methods=["POST"])
    async def dashboard_revoke(request: Request):
        form = await request.form()
        user = current_user(request)
        if not user or not csrf_ok(request, form):
            return RedirectResponse("/login", status_code=302)
        db().revoke_by_hash(form.get("h") or "", user["id"])
        return RedirectResponse("/dashboard", status_code=303)

    @mcp.custom_route("/dashboard/disconnect", methods=["POST"])
    async def dashboard_disconnect(request: Request):
        form = await request.form()
        user = current_user(request)
        if not user or not csrf_ok(request, form):
            return RedirectResponse("/login", status_code=302)
        try:
            await pool.drop(user["id"], logout=True)
        except Exception as ex:
            log.warning("logout failed for %s: %s", user["id"], ex)
        if on_disconnect is not None:
            with contextlib.suppress(Exception):
                on_disconnect(user["tg_user_id"])
        db().delete_user(user["id"])
        resp = page("قطع شد", "<h1>اتصال قطع شد</h1><p>نشست تلگرام از این سرور پاک شد.</p><a class='btn' href='/'>خانه</a>")
        clear_cookie(resp)
        return resp

    @mcp.custom_route("/logout", methods=["POST"])
    async def logout(request: Request):
        form = await request.form()
        if not csrf_ok(request, form):
            return RedirectResponse("/", status_code=302)
        resp = RedirectResponse("/", status_code=303)
        clear_cookie(resp)
        return resp
