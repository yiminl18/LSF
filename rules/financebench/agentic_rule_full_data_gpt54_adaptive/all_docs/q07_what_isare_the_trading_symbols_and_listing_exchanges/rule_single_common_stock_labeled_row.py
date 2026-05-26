import re


_TITLE_HEADER_RE = re.compile(r"\btitle\s+of\s+each\s+class\b", re.IGNORECASE)
_SYMBOL_HEADER_RE = re.compile(
    r"\btrading\s+symbol(?:\(s\))?\b|\bsymbol(?:\(s\))?\b|^trading$",
    re.IGNORECASE,
)
_EXCHANGE_HEADER_RE = re.compile(
    r"\bname\s+of\s+(?:each\s+)?exchange(?:\s+on\s+which\s+registered)?\b"
    r"|\bon\s+which\s+registered\b|^registered$",
    re.IGNORECASE,
)
_END_RE = re.compile(
    r"indicate\s+by\s+check\s+mark"
    r"|emerging\s+growth\s+company"
    r"|securities\s+registered\s+pursuant\s+to\s+section\s*12\s*\(\s*g\s*\)",
    re.IGNORECASE,
)
_COMMON_STOCK_RE = re.compile(r"\bcommon\s+stock\b|\bordinary\s+shares?\b", re.IGNORECASE)
_OTHER_SECURITY_RE = re.compile(r"\bnotes?\s+due\b|\bpreferred\b|\bdepositary\b", re.IGNORECASE)
_SYMBOL_RE = re.compile(r"^(?:[A-Z0-9][A-Z0-9.\-\/]{0,20}|—|–|--|-)$")
_EXCHANGE_RE = re.compile(
    r"\b(?:"
    r"New\s+York\s+Stock\s+Exchange(?:,\s*Inc\.)?"
    r"|The\s+Nasdaq\s+Stock\s+Market\s+LLC"
    r"|The\s+Nasdaq\s+Global\s+Select\s+Market"
    r"|Nasdaq(?:\s+Global\s+Select\s+Market|\s+Global\s+Market|\s+Capital\s+Market|\s+National\s+Market)?"
    r"|NASDAQ(?:\s+Global\s+Select\s+Market|\s+Global\s+Market|\s+Capital\s+Market|\s+National\s+Market)?"
    r"|NYSE(?:\s+American|\s+Arca)?"
    r"|Chicago\s+Stock\s+Exchange(?:,\s*Inc\.)?"
    r")\b",
    re.IGNORECASE,
)


def rule_single_common_stock_labeled_row(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            text = (text or "").replace("\u00a0", " ")
            return re.sub(r"\s+", " ", text).strip()

        def is_header(text: str) -> bool:
            cleaned = norm(text)
            return bool(cleaned) and (
                bool(_TITLE_HEADER_RE.search(cleaned))
                or bool(_SYMBOL_HEADER_RE.search(cleaned))
                or bool(_EXCHANGE_HEADER_RE.search(cleaned))
            )

        def is_symbol(text: str) -> bool:
            cleaned = norm(text)
            return bool(cleaned) and bool(_SYMBOL_RE.match(cleaned)) and not bool(_EXCHANGE_RE.search(cleaned))

        def is_exchange(text: str) -> bool:
            return bool(_EXCHANGE_RE.search(norm(text)))

        non_empty = [item for item in (doc.get("lines") or []) if isinstance(item, dict) and norm(item.get("text") or "")]
        if not non_empty:
            return []

        start_idx = None
        max_scan = min(len(non_empty), 140)
        for idx in range(max_scan):
            window = [norm(item.get("text") or "") for item in non_empty[idx : min(max_scan, idx + 6)]]
            has_title = any(_TITLE_HEADER_RE.search(text) for text in window)
            has_symbol = any(_SYMBOL_HEADER_RE.search(text) for text in window)
            has_exchange = any(_EXCHANGE_HEADER_RE.search(text) for text in window)
            if has_title and has_symbol and has_exchange:
                start_idx = idx
                break

        if start_idx is None:
            return []

        block = []
        for item in non_empty[start_idx:]:
            cleaned = norm(item.get("text") or "")
            if _END_RE.search(cleaned):
                break
            if is_header(cleaned):
                continue
            block.append(item)

        titles = [norm(item.get("text") or "") for item in block if _COMMON_STOCK_RE.search(item.get("text") or "")]
        if len(titles) != 1:
            return []
        if any(_OTHER_SECURITY_RE.search(norm(item.get("text") or "")) for item in block):
            return []

        title_idx = None
        for idx, item in enumerate(block):
            if norm(item.get("text") or "") == titles[0]:
                title_idx = idx
                break
        if title_idx is None:
            return []

        symbol = ""
        exchange = ""
        for idx in range(title_idx + 1, min(len(block), title_idx + 5)):
            candidate = norm(block[idx].get("text") or "")
            if not symbol and is_symbol(candidate):
                symbol = candidate
                continue
            if is_exchange(candidate):
                exchange = candidate
                break

        if not (title_idx is not None and symbol and exchange):
            return []

        source = block[title_idx]
        span = {
            "text": f"Title of each class: {titles[0]}; Trading Symbol(s): {symbol}; Name of each exchange on which registered: {exchange}",
        }
        if source.get("page_no") is not None:
            span["page_no"] = source.get("page_no")
        if source.get("line_no") is not None:
            span["line_no"] = source.get("line_no")
        return [span]
    except Exception:
        return []
