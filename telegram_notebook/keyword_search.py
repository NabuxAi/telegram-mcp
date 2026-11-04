"""Query tokenization for lexical (keyword) search.

The keyword fallback matches a query against stored chunk text. Matching the whole
query as one ``LIKE '%...%'`` substring means a natural-language question such as
"فرشاد کیه؟" never matches (no message contains that exact phrase), so queries are
split into terms and matched term-by-term, ranked by how many terms a chunk hits.

Pure, dependency-free, and Unicode-aware (Persian/Arabic letters are word chars),
so it is trivially unit-testable.
"""

from __future__ import annotations

import re

# Runs of Unicode word characters, excluding underscore — splits on whitespace and
# punctuation (؟ ? ، . ! …) so question marks and separators don't bleed into a term.
_TERM_RE = re.compile(r"[^\W_]+", re.UNICODE)


def query_terms(query: str, *, min_len: int = 2, max_terms: int = 12) -> list[str]:
    """Split a search query into lowercased, de-duplicated terms.

    Drops terms shorter than ``min_len`` (one-letter noise) and caps the count so a
    pathological query can't explode the generated SQL. Returns terms in first-seen
    order; an empty list means the caller should fall back to whole-query matching.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for match in _TERM_RE.findall(query.lower()):
        if len(match) < min_len or match in seen:
            continue
        seen.add(match)
        terms.append(match)
        if len(terms) >= max_terms:
            break
    return terms
