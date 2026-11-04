from telegram_notebook.bot import NotebookBot, normalize_code, normalize_phone


def test_normalize_phone_basic_and_persian():
    assert normalize_phone("09123456789") == "+09123456789"
    assert normalize_phone("۰۹۱۲۳۴۵۶۷۸۹") == "+09123456789"
    assert normalize_phone("+98 912 345 6789") == "+989123456789"


def test_normalize_phone_rejects_out_of_range():
    assert normalize_phone("12345") is None
    assert normalize_phone("1234567890123456") is None


def test_normalize_code_variants():
    assert normalize_code("12345") == "12345"
    assert normalize_code("1-2-3 4-5") == "12345"
    assert normalize_code("۱۲۳۴۵") == "12345"


def test_normalize_code_bounds():
    assert normalize_code("12") is None
    assert normalize_code("1234567890123") is None
    assert normalize_code("") is None


def test_split_filters():
    assert NotebookBot._split_filters("q --source https://t.me/x") == ("q", "https://t.me/x", None)
    assert NotebookBot._split_filters("  just a query ") == ("just a query", None, None)
    assert NotebookBot._split_filters("") == ("", None, None)
    # --tag captures the rest of the line (may contain spaces)
    assert NotebookBot._split_filters("q --tag AI Tools") == ("q", None, "AI Tools")
    # both filters together, in either order
    assert NotebookBot._split_filters("q --source https://t.me/x --tag AI Tools") == ("q", "https://t.me/x", "AI Tools")
    assert NotebookBot._split_filters("q --tag AI Tools --source https://t.me/x") == ("q", "https://t.me/x", "AI Tools")
