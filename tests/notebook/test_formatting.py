import telegram_notebook.formatting as fmt
from telegram_notebook.formatting import to_telegram_markdown


def test_preserves_literal_content():
    # Whether or not telegramify-markdown is installed, the words survive.
    rendered, mode = to_telegram_markdown("**Bold** and `code`")
    assert "Bold" in rendered and "code" in rendered
    assert mode in ("MarkdownV2", None)


def test_falls_back_to_plain_when_library_missing(monkeypatch):
    monkeypatch.setattr(fmt, "_markdownify", None)
    rendered, mode = to_telegram_markdown("**hi** <b>x</b>")
    assert rendered == "**hi** <b>x</b>"
    assert mode is None


def test_falls_back_to_plain_on_conversion_error(monkeypatch):
    def boom(_text):
        raise ValueError("bad markdown")

    monkeypatch.setattr(fmt, "_markdownify", boom)
    rendered, mode = to_telegram_markdown("some text")
    assert rendered == "some text"
    assert mode is None
