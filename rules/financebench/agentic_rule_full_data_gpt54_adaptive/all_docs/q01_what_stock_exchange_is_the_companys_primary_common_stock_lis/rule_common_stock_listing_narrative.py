import re


_EQUITY_RE = re.compile(
    r"\b(?:common\s+(?:stock|shares?)|ordinary\s+shares?|class\s+[A-Z]\s+common\s+stock)\b",
    re.IGNORECASE,
)
_LISTING_RE = re.compile(
    r"\b(?:listed\s+on|traded\s+on|currently\s+listed\s+on|currently\s+trades\s+on|principal\s+market)\b",
    re.IGNORECASE,
)
_TABLE_MARKER_RE = re.compile(
    r"Securities\s+registered\s+pursuant\s+to\s+Section\s*12\s*\(\s*b\s*\)|"
    r"Name\s+of\s+each\s+exchange\s+on\s+which\s+registered",
    re.IGNORECASE,
)
_SECONDARY_RE = re.compile(r"\b(?:also\s+traded\s+on|also\s+listed\s+on)\b", re.IGNORECASE)
_BRACKETED_RE = re.compile(
    r"\[\s*(NYSE|NASDAQ|Nasdaq|NYSE\s+American|NYSE\s+Arca)\s*:\s*[A-Z0-9.\-]+\s*\]",
    re.IGNORECASE,
)


def rule_common_stock_listing_narrative(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "").replace("\u00a0", " ")).strip()

        def exchange_patterns() -> list[str]:
            return [
                r"The\s+NASDAQ\s+Stock\s+Market\s+LLC\s*\(\s*NASDAQ\s+Global\s+Select\s+Market\s*\)",
                r"New\s+York\s+Stock\s+Exchange\s*\(\s*NYSE\s*\)",
                r"The\s+NASDAQ\s+Global\s+Select\s+Market",
                r"The\s+Nasdaq\s+Global\s+Select\s+Market",
                r"The\s+NASDAQ\s+Stock\s+Market\s+LLC",
                r"The\s+Nasdaq\s+Stock\s+Market\s+LLC",
                r"NASDAQ\s+Global\s+Select\s+Market",
                r"Nasdaq\s+Global\s+Select\s+Market",
                r"NASDAQ\s+Global\s+Market",
                r"Nasdaq\s+Global\s+Market",
                r"NASDAQ\s+Capital\s+Market",
                r"Nasdaq\s+Capital\s+Market",
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

        def ordered_items() -> list[dict]:
            lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
            if lines:
                return sorted(
                    lines,
                    key=lambda item: (
                        int(item.get("page_no") or 0),
                        int(item.get("line_no") or 0),
                        int(item.get("paragraph_no") or 0),
                    ),
                )[:250]
            paragraphs = [item for item in (doc.get("paragraphs") or []) if isinstance(item, dict)]
            return sorted(
                paragraphs,
                key=lambda item: (
                    int(item.get("page_no") or 0),
                    int(item.get("paragraph_no") or 0),
                        int(item.get("line_no") or 0),
                ),
            )[:120]

        def table_candidate_exists(items: list[dict]) -> bool:
            for item in items[:80]:
                text = norm(item.get("text") or "")
                if _TABLE_MARKER_RE.search(text):
                    return True
            return False

        items = ordered_items()
        if not items:
            return []
        if table_candidate_exists(items):
            return []

        best: tuple[int, str, dict] | None = None
        for idx, item in enumerate(items):
            text = norm(item.get("text") or "")
            if not text:
                continue
            lower = text.lower()
            if "preferred stock" in lower:
                continue

            candidate_text = text
            if idx + 1 < len(items):
                next_text = norm(items[idx + 1].get("text") or "")
                if next_text.startswith("(") and extract_exchange(text + " " + next_text):
                    candidate_text = norm(text + " " + next_text)

            exchange_text = extract_exchange(candidate_text)
            if not exchange_text:
                continue

            score = 0
            if _EQUITY_RE.search(candidate_text):
                score += 4
            if _LISTING_RE.search(candidate_text):
                score += 4
            if _BRACKETED_RE.search(candidate_text):
                score += 6
            if _SECONDARY_RE.search(candidate_text):
                score -= 6
            if "title of each class" in lower or "trading symbol" in lower:
                score -= 4

            window_text = candidate_text
            for j in range(max(0, idx - 2), min(len(items), idx + 3)):
                if j == idx:
                    continue
                window_text += " " + norm(items[j].get("text") or "")
            if _EQUITY_RE.search(window_text):
                score += 2
            if _LISTING_RE.search(window_text):
                score += 2
            if _SECONDARY_RE.search(window_text):
                score -= 4

            if best is None or score > best[0]:
                best = (score, exchange_text, item)

        if best is None or best[0] <= 0:
            return []

        _, exchange_text, src = best
        span = {"text": exchange_text}
        if src.get("page_no") is not None:
            span["page_no"] = src.get("page_no")
        if src.get("line_no") is not None:
            span["line_no"] = src.get("line_no")
        if src.get("paragraph_no") is not None:
            span["paragraph_no"] = src.get("paragraph_no")
        return [span]
    except Exception:
        return []
