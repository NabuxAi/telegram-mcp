from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class SearchResult:
    score: float
    channel_title: str | None = None
    channel_url: str = ""
    message_url: str | None = None
    media_kind: str = ""
    file_name: str | None = None
    chunk_text: str = ""
    caption: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
