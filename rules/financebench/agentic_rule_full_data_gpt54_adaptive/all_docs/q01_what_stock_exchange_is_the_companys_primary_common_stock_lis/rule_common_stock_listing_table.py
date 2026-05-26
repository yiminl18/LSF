import re


_SECTION_RE = re.compile(
    r"Securities\s+registered\s+pursuant\s+to\s+Section\s*12\s*\(\s*b\s*\)(?:\s+of\s+the\s+Act)?",
    re.IGNORECASE,
)
_HEADER_RE = re.compile(
    r"Title\s+of\s+each\s+class|Trading\s+Symbol(?:\(s\))?|Name\s+of\s+each\s+exchange\s+on\s+which\s+registered",
    re.IGNORECASE,
)
_END_RE = re.compile(
    r"Securities\s+registered\s+pursuant\s+to\s+Section\s*12\s*\(\s*g\s*\)(?:\s+of\s+the\s+Act)?"
    r"|Indicate\s+by\s+check\s+mark"
    r"|If\s+an\s+emerging\s+growth\s+company"
    r"|The\s+aggregate\s+market\s+value"
    r"|The\s+number\s+of\s+shares\s+outstanding"
    r"|Outstanding\s+at"
    r"|As\s+of\s+.*shares\s+of\s+common\s+stock"
    r"|Item\s+1\.",
    re.IGNORECASE,
)
_EQUITY_RE = re.compile(
    r"\b(?:common\s+(?:stock|shares?)|ordinary\s+shares?|class\s+[A-Z]\s+common\s+stock)\b",
    re.IGNORECASE,
)
_SECONDARY_RE = re.compile(r"\b(?:also\s+traded\s+on|also\s+listed\s+on)\b", re.IGNORECASE)
_BRACKETED_RE = re.compile(
    r"\[\s*(NYSE|NASDAQ|Nasdaq|NYSE\s+American|NYSE\s+Arca)\s*:\s*[A-Z0-9.\-]+\s*\]",
    re.IGNORECASE,
)


def rule_common_stock_listing_table(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "").replace("\u00a0", " ")).strip()

        def ordered_items() -> list[dict]:
            items = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
            if not items:
                items = [item for item in (doc.get("paragraphs") or []) if isinstance(item, dict)]
            ordered = sorted(
                items,
                key=lambda item: (
                    int(item.get("page_no") or 0),
                    int(item.get("line_no") or 0),
                    int(item.get("paragraph_no") or 0),
                ),
            )
            return ordered[:300]

        def exchange_patterns() -> list[str]:
            return [
                r"The\s+NASDAQ\s+Stock\s+Market\s+LLC\s*\(\s*NASDAQ\s+Global\s+Select\s+Market\s*\)",
                r"New\s+York\s+Stock\s+Exchange\s*\(\s*NYSE\s*\)",
                r"The\s+NASDAQ\s+Global\s+Select\s+Market",
                r"The\s+Nasdaq\s+Global\s+Select\s+Market",
                r"The\s+NASDAQ\s+Stock\s+Market\s+LLC",
                r"The\s+Nasdaq\s+Stock\s+Market\s+LLC",
                r"The\s+NASDAQ\s+Stock\s+Market\s+LLC\s*\(\s*NASDAQ\s+Global\s+Market\s*\)",
                r"The\s+Nasdaq\s+Stock\s+Market\s+LLC\s*\(\s*Nasdaq\s+Global\s+Market\s*\)",
                r"NASDAQ\s+Global\s+Select\s+Market",
                r"Nasdaq\s+Global\s+Select\s+Market",
                r"NASDAQ\s+Global\s+Market",
                r"Nasdaq\s+Global\s+Market",
                r"NASDAQ\s+Capital\s+Market",
                r"Nasdaq\s+Capital\s+Market",
                r"The\s+NASDAQ\s+Stock\s+Market\s+LLC",
                r"The\s+Nasdaq\s+Stock\s+Market\s+LLC",
                r"New\s+York\s+Stock\s+Exchange(?:,?\s+Inc\.)?(?:\s+LLC)?",
                r"The\s+New\s+York\s+Stock\s+Exchange",
                r"NYSE",
                r"NASDAQ",
            ]

        def extract_exchange(text: str) -> str | None:
            cleaned = norm(text)
            if not cleaned:
                return None
            bracketed = _BRACKETED_RE.search(cleaned)
            if bracketed:
                return norm(bracketed.group(1))
            for pattern in exchange_patterns():
                match = re.search(pattern, cleaned, re.IGNORECASE)
                if match:
                    return norm(match.group(0))
            return None

        def maybe_join(items: list[dict], idx: int) -> str:
            current = norm(items[idx].get("text") or "")
            if not current:
                return ""
            pieces = [current]
            if idx + 1 < len(items):
                nxt = norm(items[idx + 1].get("text") or "")
                if nxt.startswith("(") and extract_exchange(current + " " + nxt):
                    pieces.append(nxt)
            return norm(" ".join(pieces))

        def add_span(text: str, src: dict) -> list[dict]:
            span = {"text": norm(text)}
            if src.get("page_no") is not None:
                span["page_no"] = src.get("page_no")
            if src.get("line_no") is not None:
                span["line_no"] = src.get("line_no")
            if src.get("paragraph_no") is not None:
                span["paragraph_no"] = src.get("paragraph_no")
            return [span]

        items = ordered_items()
        if not items:
            return []

        for idx, item in enumerate(items):
            text = norm(item.get("text") or "")
            if not text:
                continue
            if not (_SECTION_RE.search(text) or _HEADER_RE.search(text) or _EQUITY_RE.search(text)):
                continue

            section_start = max(0, idx - 2)
            section_end = min(len(items), idx + 40)
            for j in range(idx + 1, min(len(items), idx + 80)):
                if _END_RE.search(norm(items[j].get("text") or "")):
                    section_end = j
                    break
            block = items[section_start:section_end]
            if not block:
                continue

            best: tuple[int, str, dict] | None = None
            for j, current in enumerate(block):
                current_text = maybe_join(block, j)
                if not current_text:
                    continue

                exchange_text = extract_exchange(current_text)
                score = 0
                lower = current_text.lower()

                if _SECTION_RE.search(current_text):
                    score += 4
                if _HEADER_RE.search(current_text):
                    score += 3
                if _EQUITY_RE.search(current_text):
                    score += 4
                if exchange_text:
                    score += 6
                if _SECONDARY_RE.search(current_text):
                    score -= 6

                if not exchange_text:
                    for k in range(max(0, j - 4), min(len(block), j + 6)):
                        candidate_text = maybe_join(block, k)
                        candidate_exchange = extract_exchange(candidate_text)
                        if not candidate_exchange:
                            continue
                        exchange_text = candidate_exchange
                        score += 3
                        if k > j and _EQUITY_RE.search(current_text):
                            score += 3
                        neighbor_text = current_text + " " + candidate_text
                        if _SECONDARY_RE.search(neighbor_text):
                            score -= 6
                        break

                if not exchange_text:
                    continue
                if "preferred stock" in lower and not _EQUITY_RE.search(current_text):
                    continue

                if best is None or score > best[0]:
                    best = (score, exchange_text, current)

            if best is not None:
                _, exchange_text, src = best
                return add_span(exchange_text, src)

        return []
    except Exception:
        return []
