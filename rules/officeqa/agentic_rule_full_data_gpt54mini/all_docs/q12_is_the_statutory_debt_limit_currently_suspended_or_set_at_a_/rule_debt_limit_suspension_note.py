import re


_CURRENT_SUSPENSION_RE = re.compile(
    r"(?:"
    r"treasury[’']s borrowing limit was suspended until early 2025"
    r"|borrowing limit was suspended until early 2025"
    r"|treasury[’']s borrowing limit was suspended until 2025"
    r"|borrowing limit was suspended until 2025"
    r"|statutory debt limit was suspended through january 1, 2025"
    r"|fiscal responsibility act of 2023[^.]{0,120}suspended through january 1, 2025"
    r")",
    re.IGNORECASE,
)


def rule_debt_limit_suspension_note(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def add_span(spans: list[dict], seen: set[str], item: dict) -> None:
            text = norm(item.get("text") or "")
            if not text:
                return
            key = text.lower()
            if key in seen:
                return
            if not _CURRENT_SUSPENSION_RE.search(text):
                return
            seen.add(key)
            span = {"text": text}
            if item.get("page_no") is not None:
                span["page_no"] = item.get("page_no")
            if item.get("paragraph_no") is not None:
                span["paragraph_no"] = item.get("paragraph_no")
            if item.get("line_no") is not None:
                span["line_no"] = item.get("line_no")
            spans.append(span)

        spans: list[dict] = []
        seen: set[str] = set()

        for item in (doc.get("paragraphs") or []):
            if isinstance(item, dict):
                add_span(spans, seen, item)

        if spans:
            return spans

        for item in (doc.get("lines") or []):
            if isinstance(item, dict):
                add_span(spans, seen, item)

        return spans
    except Exception:
        return []
