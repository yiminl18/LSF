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
    r"Chicago\s+Stock\s+Exchange(?:,?\s+Inc\.)?|"
    r"London\s+Stock\s+Exchange|"
    r"Toronto\s+Stock\s+Exchange|"
    r"Australian\s+Securities\s+Exchange|"
    r"Hong\s+Kong\s+Stock\s+Exchange|"
    r"Euronext(?:\s+[A-Z][A-Za-z]+)?"
    r")\b",
    re.IGNORECASE,
)


def _extract_exchange(text: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", (text or "").replace("\u00a0", " ")).strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    bracket = re.search(r"\[(nyse|nasdaq|nyse american|nyse arca)\s*:", lowered)
    if bracket:
        token = bracket.group(1).upper()
        if token == "NYSE":
            return "NYSE"
        if token == "NASDAQ":
            return "NASDAQ"
        if token == "NYSE AMERICAN":
            return "NYSE American"
        if token == "NYSE ARCA":
            return "NYSE Arca"
    for pattern in [
        r"new york stock exchange(?:,\s*inc\.)?",
        r"nyse\s+american",
        r"nyse\s+arca",
        r"the\s+nasdaq\s+stock\s+market\s+llc",
        r"nasdaq(?:\s+global\s+select\s+market|\s+global\s+market|\s+capital\s+market)?",
        r"six\s+swiss\s+exchange",
        r"chicago\s+stock\s+exchange(?:,\s*inc\.)?",
        r"london\s+stock\s+exchange",
        r"toronto\s+stock\s+exchange",
        r"australian\s+securities\s+exchange",
        r"hong\s+kong\s+stock\s+exchange",
        r"euronext(?:\s+[a-z][a-z]+)?",
    ]:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(0)).strip()
    return None


def rule_common_stock_listing_table(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "").replace("\u00a0", " ")).strip()

        def ordered_items() -> list[dict]:
            items = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
            if not items:
                items = [item for item in (doc.get("paragraphs") or []) if isinstance(item, dict)]
            return sorted(
                items,
                key=lambda item: (
                    int(item.get("page_no") or 0),
                    int(item.get("line_no") or 0),
                    int(item.get("paragraph_no") or 0),
                ),
            )

        def extract_exchange(text: str) -> str | None:
            cleaned = norm(text)
            if not cleaned:
                return None
            return _extract_exchange(cleaned)

        def add_span(spans: list[dict], seen: set[str], text: str, src: dict) -> None:
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

        items = ordered_items()
        if not items:
            return []

        spans: list[dict] = []
        seen: set[str] = set()

        for idx, item in enumerate(items):
            text = norm(item.get("text") or "")
            if not text:
                continue
            if not (_SECTION_RE.search(text) or _EQUITY_RE.search(text) or "name of each exchange on which registered" in text.lower()):
                continue

            section_start = idx if _SECTION_RE.search(text) else max(0, idx - 2)
            section_end = len(items)
            for j in range(idx + 1, len(items)):
                if _END_RE.search(norm(items[j].get("text") or "")):
                    section_end = j
                    break
            block = items[section_start : min(len(items), section_end)]
            if not block:
                continue

            best: tuple[int, dict, str] | None = None
            for j, current in enumerate(block):
                current_text = norm(current.get("text") or "")
                if not current_text:
                    continue
                exchange = extract_exchange(current_text)
                score = 0
                if _SECTION_RE.search(current_text):
                    score += 5
                if "name of each exchange on which registered" in current_text.lower():
                    score += 4
                if "title of each class" in current_text.lower():
                    score += 3
                if _EQUITY_RE.search(current_text):
                    score += 3
                if exchange:
                    score += 6
                if exchange and _EQUITY_RE.search(current_text):
                    score += 3
                if "also traded on" in current_text.lower() or "also listed on" in current_text.lower():
                    score -= 5
                if current_text.lower().startswith("note:") and "title of each class" not in current_text.lower():
                    score -= 2
                if not exchange and not _EQUITY_RE.search(current_text) and "trading symbol" not in current_text.lower():
                    continue
                window_start = max(0, j - 3)
                window_end = min(len(block), j + 4)
                if not exchange:
                    for k in range(window_start, window_end):
                        candidate = block[k]
                        candidate_text = norm(candidate.get("text") or "")
                        exchange = extract_exchange(candidate_text)
                        if exchange:
                            score += 2
                            break
                if exchange:
                    candidate_src = current
                    if best is None or score > best[0]:
                        best = (score, candidate_src, exchange)

            if best is not None:
                _, src, exchange_text = best
                add_span(spans, seen, exchange_text, src)
                return spans

        return spans
    except Exception:
        return []
