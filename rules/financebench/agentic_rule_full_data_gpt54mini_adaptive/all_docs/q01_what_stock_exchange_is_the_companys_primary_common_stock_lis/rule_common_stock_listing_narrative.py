import re


_EQUITY_RE = re.compile(
    r"\b(?:common\s+(?:stock|shares)|ordinary\s+shares?|class\s+[A-Z]\s+common\s+stock)\b",
    re.IGNORECASE,
)
_PRIMARY_RE = re.compile(r"\b(?:primary\s+exchange|principal\s+market)\b", re.IGNORECASE)
_LISTING_RE = re.compile(
    r"\b(?:"
    r"listed\s+on|"
    r"traded\s+on|"
    r"currently\s+listed\s+on|"
    r"currently\s+trades\s+on|"
    r"stock\s+is\s+listed\s+on|"
    r"common\s+stock\s+is\s+listed\s+on|"
    r"common\s+stock\s+is\s+traded\s+on|"
    r"principal\s+market\s+for\s+(?:our|the)\s+common\s+stock\s+is|"
    r"our\s+common\s+stock\s+is\s+listed\s+on|"
    r"our\s+common\s+stock\s+is\s+traded\s+on"
    r")\b",
    re.IGNORECASE,
)
_TABLE_HEADER_RE = re.compile(
    r"Securities\s+registered\s+pursuant\s+to\s+Section\s*12\s*\(\s*b\s*\)(?:\s+of\s+the\s+Act)?",
    re.IGNORECASE,
)
_TABLE_END_RE = re.compile(
    r"Securities\s+registered\s+pursuant\s+to\s+Section\s*12\s*\(\s*g\s*\)(?:\s+of\s+the\s+Act)?"
    r"|Indicate\s+by\s+check\s+mark"
    r"|If\s+securities\s+are\s+registered\s+pursuant\s+to\s+Section\s*12\s*\(\s*b\s*\)",
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
_BRACKETED_EXCHANGE_RE = re.compile(
    r"\[(?:NYSE|NASDAQ|Nasdaq|NYSE\s+American|NYSE\s+Arca)\s*:\s*[A-Z0-9.\-]+\]",
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


def rule_common_stock_listing_narrative(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "").replace("\u00a0", " ")).strip()

        def extract_exchange(text: str) -> str | None:
            cleaned = norm(text)
            if not cleaned:
                return None
            return _extract_exchange(cleaned)

        def score(text: str) -> int:
            cleaned = norm(text)
            if not cleaned:
                return 0
            lower = cleaned.lower()
            if "preferred stock" in lower:
                return 0
            score_val = 0
            if _EQUITY_RE.search(cleaned):
                score_val += 2
            if _LISTING_RE.search(cleaned):
                score_val += 3
            if _PRIMARY_RE.search(cleaned):
                score_val += 2
            if _EXCHANGE_RE.search(cleaned):
                score_val += 2
            if _BRACKETED_EXCHANGE_RE.search(cleaned):
                score_val += 4
            if "notes" in lower and "common stock" not in lower:
                score_val -= 2
            return score_val

        def table_candidate_exists(items: list[dict]) -> bool:
            ordered = sorted(
                items,
                key=lambda item: (
                    int(item.get("page_no") or 0),
                    int(item.get("line_no") or 0),
                    int(item.get("paragraph_no") or 0),
                ),
            )
            for idx, item in enumerate(ordered):
                text = norm(item.get("text") or "")
                if not text or not _TABLE_HEADER_RE.search(text):
                    continue
                section_end = len(ordered)
                for j in range(idx + 1, len(ordered)):
                    if _TABLE_END_RE.search(norm(ordered[j].get("text") or "")):
                        section_end = j
                        break
                for j in range(idx + 1, min(section_end, idx + 24)):
                    current = norm(ordered[j].get("text") or "")
                    if current and _EQUITY_RE.search(current) and _EXCHANGE_RE.search(current):
                        return True
                    if current and _EQUITY_RE.search(current):
                        for k in range(j, min(section_end, j + 8)):
                            if _EXCHANGE_RE.search(norm(ordered[k].get("text") or "")):
                                return True
            return False

        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        paragraphs = [item for item in (doc.get("paragraphs") or []) if isinstance(item, dict)]
        items = lines or paragraphs
        if not items:
            return []

        if lines and table_candidate_exists(lines):
            return []

        spans: list[dict] = []
        seen: set[str] = set()

        def add(src: dict, text: str) -> None:
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

        ranked: list[tuple[int, dict, str]] = []
        for item in items:
            text = norm(item.get("text") or "")
            s = score(text)
            exchange = extract_exchange(text)
            if exchange:
                if "also traded on" in text.lower() or "also listed on" in text.lower():
                    s -= 5
                ranked.append((s + 4, item, exchange))
                continue
            if s > 0:
                ranked.append((s, item, text))

        ranked.sort(
            key=lambda x: (
                -x[0],
                len(x[2]),
                int(x[1].get("page_no") or 0),
                int(x[1].get("line_no") or 0),
                int(x[1].get("paragraph_no") or 0),
            )
        )

        if ranked:
            _, item, text = ranked[0]
            exchange = extract_exchange(text)
            add(item, exchange or text)

        return spans
    except Exception:
        return []
