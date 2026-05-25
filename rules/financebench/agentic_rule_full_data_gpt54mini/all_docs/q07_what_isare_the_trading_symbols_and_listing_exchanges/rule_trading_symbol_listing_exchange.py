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
_HEADER_RE = re.compile(
    r"\bTitle\s+of\s+each\s+class\b"
    r"|\bTrading(?:\s+symbol(?:\(s\))?)?\b"
    r"|\bsymbol(?:\(s\))?\b"
    r"|\bName\s+of\s+(?:each\s+)?exchange(?:\s+on\s+which\s+registered)?\b"
    r"|\bon\s+which\s+registered\b"
    r"|\bSecurities\s+registered\s+pursuant\s+to\s+Section\s*12\s*\(\s*b\s*\)",
    re.IGNORECASE,
)
_SYMBOL_RE = re.compile(
    r"^(?:[A-Z0-9][A-Z0-9.\-\/]{0,20}|—|–|--|-)$"
)
_EXCHANGE_RE = re.compile(
    r"\b(?:New\s+York\s+Stock\s+Exchange|NYSE"
    r"|Nasdaq|NASDAQ"
    r"|Stock\s+Market"
    r"|Stock\s+Exchange"
    r"|Global\s+Select\s+Market"
    r"|Global\s+Market"
    r"|NYSE\s+American"
    r"|NYSE\s+Arca)\b",
    re.IGNORECASE,
)


def rule_trading_symbol_listing_exchange(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "").replace("\u00a0", " ")).strip()

        def is_header(text: str) -> bool:
            cleaned = norm(text)
            return bool(cleaned) and bool(_HEADER_RE.search(cleaned))

        def is_symbol(text: str) -> bool:
            cleaned = norm(text)
            if not cleaned:
                return False
            if _EXCHANGE_RE.search(cleaned):
                return False
            if any(ch.islower() for ch in cleaned):
                return False
            return bool(_SYMBOL_RE.match(cleaned))

        def is_exchange(text: str) -> bool:
            return bool(_EXCHANGE_RE.search(norm(text)))

        def is_terminator(text: str) -> bool:
            cleaned = norm(text)
            return bool(cleaned) and bool(_END_RE.search(cleaned))

        def extract_from_items(items: list[dict]) -> list[dict]:
            row_texts: list[str] = []
            first_source: dict | None = None
            seen: set[str] = set()

            filtered: list[dict] = []
            started = False
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = norm(item.get("text") or "")
                if not text:
                    continue
                if not started:
                    if _SECTION_RE.search(text):
                        started = True
                    continue
                if is_terminator(text):
                    break
                if is_header(text):
                    continue
                filtered.append(item)

            if not filtered:
                return []

            idx = 0
            while idx < len(filtered):
                item = filtered[idx]
                text = norm(item.get("text") or "")
                if not text or is_header(text):
                    idx += 1
                    continue

                symbol_idx = None
                for look_ahead in range(idx + 1, min(len(filtered), idx + 5)):
                    if is_symbol(filtered[look_ahead].get("text") or ""):
                        symbol_idx = look_ahead
                        break

                if symbol_idx is None:
                    idx += 1
                    continue

                exchange_idx = None
                for look_ahead in range(symbol_idx + 1, min(len(filtered), symbol_idx + 4)):
                    if is_exchange(filtered[look_ahead].get("text") or ""):
                        exchange_idx = look_ahead
                        break

                if exchange_idx is None:
                    idx += 1
                    continue

                symbol_item = filtered[symbol_idx]
                exchange_item = filtered[exchange_idx]
                symbol_text = norm(symbol_item.get("text") or "")
                exchange_text = norm(exchange_item.get("text") or "")
                if symbol_text:
                    if symbol_text in {"—", "–", "--", "-"}:
                        combined_text = f"{symbol_text} {exchange_text}"
                    else:
                        combined_text = f"{symbol_text} — {exchange_text}"
                    key = combined_text.lower()
                    if key not in seen:
                        seen.add(key)
                        row_texts.append(combined_text)
                        if first_source is None:
                            first_source = symbol_item
                idx = exchange_idx + 1

            if row_texts:
                span = {"text": "; ".join(row_texts)}
                if first_source:
                    if first_source.get("page_no") is not None:
                        span["page_no"] = first_source.get("page_no")
                    if first_source.get("line_no") is not None:
                        span["line_no"] = first_source.get("line_no")
                    if first_source.get("paragraph_no") is not None:
                        span["paragraph_no"] = first_source.get("paragraph_no")
                return [span]
            return []

        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        spans = extract_from_items(lines)
        if spans:
            return spans

        paragraphs = [item for item in (doc.get("paragraphs") or []) if isinstance(item, dict)]
        spans = extract_from_items(paragraphs)
        if spans:
            return spans

        full_text = doc.get("text") or ""
        if full_text:
            synthetic_items = []
            for idx, line in enumerate(full_text.splitlines(), start=1):
                synthetic_items.append({"text": line, "line_no": idx})
            spans = extract_from_items(synthetic_items)
            if spans:
                return spans

        return []
    except Exception:
        return []
