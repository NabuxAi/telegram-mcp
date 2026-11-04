from dataclasses import dataclass


@dataclass(slots=True)
class TextChunk:
    index: int
    text: str
    start_char: int
    end_char: int


def split_text(text: str, chunk_size: int, overlap: int) -> list[TextChunk]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    if overlap >= chunk_size:
        raise ValueError("chunk overlap must be smaller than chunk size")

    chunks: list[TextChunk] = []
    start = 0
    index = 0
    step = chunk_size - overlap

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk_text = normalized[start:end].strip()
        if chunk_text:
            chunks.append(
                TextChunk(
                    index=index,
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                )
            )
            index += 1
        if end >= len(normalized):
            break
        start += step

    return chunks
