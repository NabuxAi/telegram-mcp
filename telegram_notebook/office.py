"""Local text extraction for Office Open XML documents (DOCX / XLSX).

DOCX and XLSX files are ZIP archives of XML parts, so their text content can be
extracted with only the standard library — no API key, no network, and no new
dependency. This complements the Gemini-based OCR/PDF extractor
(``ExtractionService``) for the document formats Gemini multimodal does not read
natively, and means forwarded Office documents become searchable even when no
LLM key is configured.

The XML matching is done on local tag names (ignoring namespaces) so the
extractors tolerate both the transitional and strict OOXML namespace variants.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import IO
from xml.etree import ElementTree as ET

DOCX_EXTENSIONS = {".docx"}
XLSX_EXTENSIONS = {".xlsx"}
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def detect_office_kind(file_name: str | None, mime_type: str | None) -> str | None:
    """Return ``"docx"``, ``"xlsx"``, or ``None`` for a forwarded document."""
    suffix = Path(file_name or "").suffix.lower()
    mt = (mime_type or "").lower()
    if suffix in DOCX_EXTENSIONS or mt == DOCX_MIME:
        return "docx"
    if suffix in XLSX_EXTENSIONS or mt == XLSX_MIME:
        return "xlsx"
    return None


def _local(tag: str) -> str:
    """Strip the XML namespace, leaving the bare element name."""
    return tag.rsplit("}", 1)[-1]


def extract_docx_text(path: Path | IO[bytes]) -> str:
    """Extract paragraph text (including text inside tables) from a .docx file.

    ``path`` may be a filesystem path or a binary file object (anything
    ``zipfile.ZipFile`` accepts), so callers can extract straight from bytes.
    """
    with zipfile.ZipFile(path) as zf:
        with zf.open("word/document.xml") as handle:
            root = ET.parse(handle).getroot()

    lines: list[str] = []
    for para in root.iter():
        if _local(para.tag) != "p":
            continue
        text = "".join(node.text or "" for node in para.iter() if _local(node.tag) == "t")
        if text.strip():
            lines.append(text)
    return "\n".join(lines).strip()


def _read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        with zf.open("xl/sharedStrings.xml") as handle:
            root = ET.parse(handle).getroot()
    except KeyError:
        return []
    strings: list[str] = []
    for si in root.iter():
        if _local(si.tag) != "si":
            continue
        strings.append("".join(t.text or "" for t in si.iter() if _local(t.tag) == "t"))
    return strings


def _cell_text(cell: ET.Element, shared: list[str]) -> str:
    ctype = cell.get("t")
    if ctype == "s":  # shared-string index
        value = next((v for v in cell.iter() if _local(v.tag) == "v"), None)
        if value is not None and value.text is not None:
            try:
                return shared[int(value.text)]
            except (ValueError, IndexError):
                return ""
        return ""
    if ctype in ("inlineStr", "str"):  # inline string / formula string result
        return "".join(t.text or "" for t in cell.iter() if _local(t.tag) == "t") or "".join(
            v.text or "" for v in cell.iter() if _local(v.tag) == "v"
        )
    value = next((v for v in cell.iter() if _local(v.tag) == "v"), None)
    return value.text if (value is not None and value.text is not None) else ""


def extract_xlsx_text(path: Path | IO[bytes]) -> str:
    """Extract cell values from every worksheet as tab/newline-separated text.

    Accepts a filesystem path or a binary file object.
    """
    with zipfile.ZipFile(path) as zf:
        shared = _read_shared_strings(zf)
        sheet_names = sorted(
            name
            for name in zf.namelist()
            if name.startswith("xl/worksheets/") and name.endswith(".xml")
        )
        blocks: list[str] = []
        for name in sheet_names:
            with zf.open(name) as handle:
                root = ET.parse(handle).getroot()
            rows: list[str] = []
            for row in root.iter():
                if _local(row.tag) != "row":
                    continue
                cells = [_cell_text(c, shared) for c in row if _local(c.tag) == "c"]
                line = "\t".join(cells).rstrip("\t")
                if line.strip():
                    rows.append(line)
            if rows:
                blocks.append("\n".join(rows))
    return "\n\n".join(blocks).strip()


def extract_office_text(path: Path, kind: str | None = None) -> str:
    """Dispatch to the DOCX or XLSX extractor based on ``kind`` (or the suffix)."""
    kind = kind or detect_office_kind(path.name, None)
    if kind == "docx":
        return extract_docx_text(path)
    if kind == "xlsx":
        return extract_xlsx_text(path)
    raise ValueError(f"Unsupported office document: {path.name}")


def extract_office_bytes(data: bytes, kind: str) -> str:
    """Extract DOCX/XLSX text straight from in-memory bytes (e.g. a backup zip).

    ``kind`` must be ``"docx"`` or ``"xlsx"`` — there is no filename to infer from.
    """
    buffer = io.BytesIO(data)
    if kind == "docx":
        return extract_docx_text(buffer)
    if kind == "xlsx":
        return extract_xlsx_text(buffer)
    raise ValueError(f"Unsupported office document kind: {kind!r}")
