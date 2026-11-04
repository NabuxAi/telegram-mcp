import zipfile

import pytest

from telegram_notebook.office import (
    DOCX_MIME,
    XLSX_MIME,
    detect_office_kind,
    extract_docx_text,
    extract_office_text,
    extract_xlsx_text,
)

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
SS_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _make_docx(path, paragraphs):
    runs = "".join(
        "<w:p>" + "".join(f"<w:r><w:t>{seg}</w:t></w:r>" for seg in segs) + "</w:p>"
        for segs in paragraphs
    )
    document = (
        f'<?xml version="1.0"?>'
        f'<w:document xmlns:w="{W_NS}"><w:body>{runs}</w:body></w:document>'
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", document)


def _make_xlsx(path, *, shared, sheets):
    sst = (
        f'<sst xmlns="{SS_NS}">'
        + "".join(f"<si><t>{s}</t></si>" for s in shared)
        + "</sst>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("xl/sharedStrings.xml", sst)
        for i, rows in enumerate(sheets, start=1):
            body = ""
            for row in rows:
                cells = ""
                for cell in row:
                    if cell["t"] == "s":
                        cells += f'<c t="s"><v>{cell["v"]}</v></c>'
                    elif cell["t"] == "inlineStr":
                        cells += f'<c t="inlineStr"><is><t>{cell["v"]}</t></is></c>'
                    else:
                        cells += f'<c><v>{cell["v"]}</v></c>'
                body += f"<row>{cells}</row>"
            sheet = f'<worksheet xmlns="{SS_NS}"><sheetData>{body}</sheetData></worksheet>'
            zf.writestr(f"xl/worksheets/sheet{i}.xml", sheet)


# --- detection ---

def test_detect_office_kind_by_extension_and_mime():
    assert detect_office_kind("report.docx", None) == "docx"
    assert detect_office_kind("budget.xlsx", None) == "xlsx"
    assert detect_office_kind("file", DOCX_MIME) == "docx"
    assert detect_office_kind("file", XLSX_MIME) == "xlsx"
    assert detect_office_kind("notes.txt", "text/plain") is None
    assert detect_office_kind(None, None) is None


# --- docx ---

def test_extract_docx_joins_runs_and_paragraphs(tmp_path):
    path = tmp_path / "doc.docx"
    _make_docx(path, [["Hello "], ["Second ", "line"], ["  "]])
    assert extract_docx_text(path) == "Hello \nSecond line"


def test_extract_office_text_dispatches_docx(tmp_path):
    path = tmp_path / "doc.docx"
    _make_docx(path, [["Dispatched"]])
    assert extract_office_text(path) == "Dispatched"


# --- xlsx ---

def test_extract_xlsx_resolves_shared_and_inline_and_numbers(tmp_path):
    path = tmp_path / "book.xlsx"
    _make_xlsx(
        path,
        shared=["Name", "Alice"],
        sheets=[
            [
                [{"t": "s", "v": 0}, {"t": "s", "v": 1}],
                [{"t": "inlineStr", "v": "Bob"}, {"t": "n", "v": 42}],
            ]
        ],
    )
    assert extract_xlsx_text(path) == "Name\tAlice\nBob\t42"


def test_extract_xlsx_multiple_sheets_blank_separated(tmp_path):
    path = tmp_path / "multi.xlsx"
    _make_xlsx(
        path,
        shared=["one", "two"],
        sheets=[[[{"t": "s", "v": 0}]], [[{"t": "s", "v": 1}]]],
    )
    assert extract_xlsx_text(path) == "one\n\ntwo"


def test_extract_xlsx_tolerates_missing_shared_strings(tmp_path):
    path = tmp_path / "nums.xlsx"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{SS_NS}"><sheetData>'
            f"<row><c><v>1</v></c><c><v>2</v></c></row>"
            f"</sheetData></worksheet>",
        )
    assert extract_xlsx_text(path) == "1\t2"


def test_extract_office_text_rejects_unknown(tmp_path):
    path = tmp_path / "thing.bin"
    path.write_bytes(b"x")
    with pytest.raises(ValueError):
        extract_office_text(path)
