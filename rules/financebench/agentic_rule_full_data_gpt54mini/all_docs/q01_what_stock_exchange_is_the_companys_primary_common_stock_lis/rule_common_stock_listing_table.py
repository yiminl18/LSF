import re


_SECTION_RE = re.compile(
    r"Securities\s+registered\s+pursuant\s+to\s+Section\s*12\s*\(\s*b\s*\)(?:\s+of\s+the\s+Act)?",
    re.IGNORECASE,
)
_END_RE = re.compile(
    r"Securities\s+registered\s+pursuant\s+to\s+Section\s*12\s*\(\s*g\s*\)(?:\s+of\s+the\s+Act)?"
    r"|Indicate\s+by\s+check\s+mark"
    r"|If\s+securities\s+are\s+registered\s+pursuant\s+to\s+Section\s*12\s*\(\s*b\s*\)",
    re.IGNORECASE,
)
_EQUITY_RE = re.compile(
    r"\b(?:common\s+(?:stock|shares)|ordinary\s+shares?|class\s+[A-Z]\s+common\s+stock)\b",
    re.IGNORECASE,
)
_EXCHANGE_RE = re.compile(
    r"\b(?:"
    r"New\s+York\s+Stock\s+Exchange(?:,?\s+Inc\.)?|"
    r"NYSE(?:\s+American|\s+Arca)?|"
    r"The\s+Nasdaq\s+Stock\s+Market\s+LLC|"
    r"Nasdaq(?:\s+Global\s+Select\s+Market|\s+Global\s+Market|\s+Capital\s+Market)?|"
    r"NASDAQ|"
    r"SIX\s+Swiss\s+Exchange|"
    r"Chicago\s+Stock\s+Exchange|"
    r"London\s+Stock\s+Exchange|"
    r"Toronto\s+Stock\s+Exchange"
    r")\b",
    re.IGNORECASE,
)


def rule_common_stock_listing_table(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "").replace("\u00a0", " ")).strip()

        def ordered_items() -> list[dict]:
            items = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
            return sorted(
                items,
                key=lambda item: (
                    int(item.get("page_no") or 0),
                    int(item.get("line_no") or 0),
                ),
            )

        def is_exchange(text: str) -> bool:
            return bool(_EXCHANGE_RE.search(norm(text)))

        def is_equity(text: str) -> bool:
            text = norm(text)
            return bool(text) and bool(_EQUITY_RE.search(text))

        items = ordered_items()
        if not items:
            return []

        spans: list[dict] = []
        seen: set[str] = set()

        def add(text: str, src: dict) -> None:
            cleaned = norm(text)
            if not cleaned:
                return
            key = cleaned.lower()
            if key in seen:
                return
            seen.add(key)
            span = {"text": cleaned}
            if src.get("page_no") is not None:
                span["page_no"] = src.get("page_no")
            if src.get("line_no") is not None:
                span["line_no"] = src.get("line_no")
            if src.get("paragraph_no") is not None:
                span["paragraph_no"] = src.get("paragraph_no")
            spans.append(span)

        for idx, item in enumerate(items):
            text = norm(item.get("text") or "")
            if not text or not _SECTION_RE.search(text):
                continue

            section_end = len(items)
            for j in range(idx + 1, len(items)):
                if _END_RE.search(norm(items[j].get("text") or "")):
                    section_end = j
                    break

            block = items[idx + 1 : section_end]
            if not block:
                continue

            for j, current in enumerate(block):
                current_text = norm(current.get("text") or "")
                if not current_text or not is_equity(current_text):
                    continue

                exchange_text = None
                exchange_src = None
                window_start = max(0, j - 6)
                window_end = min(len(block), j + 7)
                for k in range(window_start, window_end):
                    candidate = block[k]
                    candidate_text = norm(candidate.get("text") or "")
                    if is_exchange(candidate_text):
                        exchange_text = candidate_text
                        exchange_src = candidate
                        break

                if exchange_text:
                    add(f"{current_text} — {exchange_text}", current)
                    if exchange_src is not None and exchange_src is not current:
                        add(exchange_text, exchange_src)
                    return spans
                add(current_text, current)
                return spans

        return spans
    except Exception:
        return []
