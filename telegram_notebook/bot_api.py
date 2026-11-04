from __future__ import annotations

import json
import re
import ssl
from pathlib import Path
from urllib import error, request

import certifi

#: Telegram will not send us a file larger than this via the Bot API.
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


def safe_filename(name: str | None, fallback: str = "file") -> str:
    """Reduce a Telegram-supplied file name to a single safe path segment.

    The Bot API echoes ``document.file_name`` back verbatim, so it is attacker
    controlled: ``../../.env`` escapes the temp directory and ``/etc/cron.d/x``
    discards it entirely, because ``Path(tmp) / "/abs"`` yields ``/abs``.
    """
    candidate = Path(name or "").name  # strips any directory component
    candidate = _UNSAFE_NAME.sub("_", candidate).lstrip(".")
    return candidate or fallback


class TelegramBotApi:
    def __init__(self, token: str) -> None:
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.file_base_url = f"https://api.telegram.org/file/bot{token}"
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    def call(self, method: str, payload: dict[str, object] | None = None, files: dict | None = None) -> dict[str, object]:
        if files:
            # استفاده از requests برای ارسال فایل (ساده‌تر است)
            import requests

            url = f"{self.base_url}/{method}"
            try:
                res = requests.post(url, data=payload, files=files, timeout=(10, 120))
            except requests.RequestException as exc:
                # requests puts the full URL — and therefore the bot token — in its
                # exception messages. Never let that reach a log or a chat.
                raise RuntimeError(f"{method} failed: {type(exc).__name__}") from None
            try:
                data = res.json()
            except ValueError:
                raise RuntimeError(f"{method} failed: HTTP {res.status_code} (non-JSON response)") from None
            if not data.get("ok"):
                raise RuntimeError(f"{method} failed: {data.get('description') or res.status_code}")
            return data
            
        raw = None
        if payload is not None:
            raw = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/{method}",
            data=raw,
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=90, context=self.ssl_context) as response:
                data = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            # Telegram reports its real errors in the body of a 4xx/5xx, which
            # urllib would otherwise discard along with retry_after.
            try:
                data = json.loads(exc.read().decode("utf-8") or "{}")
            except ValueError:
                data = {}
            retry_after = (data.get("parameters") or {}).get("retry_after")
            detail = data.get("description") or f"HTTP {exc.code}"
            suffix = f" (retry_after={retry_after})" if retry_after else ""
            raise RuntimeError(f"{method} failed: {detail}{suffix}") from None
        if not data.get("ok"):
            raise RuntimeError(str(data))
        return data

    def get_me(self) -> dict[str, object]:
        return self.call("getMe")

    def get_updates(self, *, offset: int | None = None, timeout: int = 30) -> list[dict[str, object]]:
        payload: dict[str, object] = {"timeout": timeout, "allowed_updates": ["message", "callback_query"]}
        if offset is not None:
            payload["offset"] = offset
        return list(self.call("getUpdates", payload).get("result", []))

    def send_message(
        self,
        chat_id: int,
        text: str,
        reply_markup: dict[str, object] | None = None,
        disable_web_page_preview: bool | None = None,
        parse_mode: str | None = "HTML",
    ) -> None:
        payload: dict[str, object] = {"chat_id": chat_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        if disable_web_page_preview is not None:
            payload["disable_web_page_preview"] = disable_web_page_preview
        self.call("sendMessage", payload)

    def send_photo(
        self,
        *,
        chat_id: int,
        photo_path: Path,
        caption: str | None = None,
    ) -> None:
        payload = {"chat_id": chat_id}
        if caption:
            payload["caption"] = caption
            payload["parse_mode"] = "HTML"
        
        with open(photo_path, "rb") as f:
            self.call("sendPhoto", payload=payload, files={"photo": f})

    def send_document(self, *, chat_id: int, document_path: Path, caption: str | None = None) -> None:
        payload: dict[str, object] = {"chat_id": chat_id}
        if caption:
            payload["caption"] = caption
            payload["parse_mode"] = "HTML"
        with open(document_path, "rb") as f:
            self.call("sendDocument", payload=payload, files={"document": f})

    def send_audio(self, *, chat_id: int, audio_path: Path, caption: str | None = None, title: str | None = None) -> None:
        payload: dict[str, object] = {"chat_id": chat_id}
        if caption:
            payload["caption"] = caption
            payload["parse_mode"] = "HTML"
        if title:
            payload["title"] = title
        with open(audio_path, "rb") as f:
            self.call("sendAudio", payload=payload, files={"audio": f})

    def get_file(self, file_id: str) -> dict[str, object]:
        """Resolve a file_id to its metadata (including ``file_path``) via getFile."""
        return dict(self.call("getFile", {"file_id": file_id}).get("result", {}))

    def download_file(self, file_path: str, dest: Path) -> Path:
        """Download a Bot API file (``file_path`` from getFile) to ``dest``.

        ``dest`` is normalised to a single file inside its own parent directory:
        callers build it from the Telegram-supplied file name, so without this a
        crafted name writes anywhere the bot user can write.
        """
        dest = Path(dest)
        directory = dest.parent.resolve()
        target = (directory / safe_filename(dest.name)).resolve()
        if target.parent != directory:
            raise ValueError("refusing to write outside the download directory")

        url = f"{self.file_base_url}/{file_path}"
        req = request.Request(url)
        written = 0
        try:
            with request.urlopen(req, timeout=120, context=self.ssl_context) as response, target.open("wb") as fh:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_DOWNLOAD_BYTES:
                        raise ValueError("downloaded file exceeds the Bot API size limit")
                    fh.write(chunk)
        except error.HTTPError as exc:
            # The file URL embeds the bot token; never let urllib quote it back.
            raise RuntimeError(f"file download failed: HTTP {exc.code}") from None
        return target

    def answer_callback_query(self, callback_query_id: str) -> None:
        self.call("answerCallbackQuery", {"callback_query_id": callback_query_id})

    def delete_message(self, chat_id: int, message_id: int) -> None:
        self.call("deleteMessage", {"chat_id": chat_id, "message_id": message_id})

    @staticmethod
    def contact_keyboard() -> dict[str, object]:
        return {
            "keyboard": [
                [
                    {
                        "text": "Share Phone Number",
                        "request_contact": True,
                    }
                ]
            ],
            "resize_keyboard": True,
            "one_time_keyboard": True,
        }

    @staticmethod
    def remove_keyboard() -> dict[str, object]:
        return {"remove_keyboard": True}

    @staticmethod
    def web_app_keyboard(text: str, url: str) -> dict[str, object]:
        """An inline keyboard with a single button that launches a Mini App."""
        return {"inline_keyboard": [[{"text": text, "web_app": {"url": url}}]]}

    def set_chat_menu_button(self, *, url: str, text: str = "Open App") -> None:
        """Set the bot-wide menu button (the ☰ next to the input) to a Mini App."""
        self.call(
            "setChatMenuButton",
            {"menu_button": {"type": "web_app", "text": text, "web_app": {"url": url}}},
        )
