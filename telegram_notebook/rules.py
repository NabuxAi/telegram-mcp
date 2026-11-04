"""Rule matching for auto-tagging ingested content.

Two kinds of rule map a criterion to a tag:

* ``keyword`` rules tag content whose text contains the keyword (case-insensitive
  substring). Cheap and deterministic, so they run on every ingest.
* ``ai`` rules describe the criterion in natural language and let an LLM decide
  whether the content matches. They are more expensive, so they only run on
  demand (``/rule apply``).

Matching and the AI prompt/parse helpers are kept as pure functions; the LLM call
itself is injected (``generate``) so everything is easy to test offline.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping


def _is_ai(rule: Mapping[str, object]) -> bool:
    return str(rule.get("kind") or "keyword").lower() == "ai"


def match_tags(text: str | None, rules: Iterable[Mapping[str, object]]) -> set[str]:
    """Return the set of tags whose *keyword* rule appears in ``text``.

    AI-kind rules are ignored here — they are evaluated via :func:`classify_ai_tags`.
    """
    lowered = (text or "").lower()
    if not lowered:
        return set()
    matched: set[str] = set()
    for rule in rules:
        if _is_ai(rule):
            continue
        keyword = str(rule.get("keyword") or "").strip().lower()
        tag = rule.get("tag")
        if keyword and tag and keyword in lowered:
            matched.add(str(tag))
    return matched


def build_classify_prompt(text: str, ai_rules: list[Mapping[str, object]], *, max_chars: int = 4000) -> str:
    """Prompt asking the model which of the AI rules' tags apply to ``text``."""
    criteria = "\n".join(
        f"- {rule['tag']}: {str(rule.get('keyword') or '').strip()}" for rule in ai_rules
    )
    snippet = (text or "").strip()[:max_chars]
    return f"""You are a strict classifier. Given a piece of content and a list of tags with
their criteria, decide which tags apply to the content.

Tags and criteria:
{criteria}

Content:
{snippet}

Reply with ONLY a comma-separated list of the tag names that apply (exactly as written
above). If none apply, reply with the single word: NONE."""


def parse_classified_tags(response: str | None, ai_rules: Iterable[Mapping[str, object]]) -> set[str]:
    """Parse the model reply, keeping only tags that are defined in ``ai_rules``."""
    valid = {str(rule["tag"]).strip().lower(): str(rule["tag"]) for rule in ai_rules}
    matched: set[str] = set()
    for token in (response or "").replace("\n", ",").split(","):
        key = token.strip().lower()
        if key and key != "none" and key in valid:
            matched.add(valid[key])
    return matched


def classify_ai_tags(
    text: str,
    ai_rules: list[Mapping[str, object]],
    *,
    generate: Callable[[str], str],
) -> set[str]:
    """Run the AI rules against ``text`` using the injected ``generate`` callable.

    ``generate`` takes a prompt and returns the model's text reply. Returns the set
    of matching tags; an empty list of rules or empty text short-circuits to no call.
    """
    if not ai_rules or not (text or "").strip():
        return set()
    prompt = build_classify_prompt(text, ai_rules)
    return parse_classified_tags(generate(prompt), ai_rules)
