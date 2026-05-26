import re


_SECTION_RE = re.compile(
    r"securities\s+registered\s+pursuant\s+to\s+section\s*12\s*\(\s*b\s*\)"
    r"(?:\s+of\s+the\s+(?:act|securities\s+exchange\s+act\s+of\s+1934))?",
    re.IGNORECASE,
)
_END_RE = re.compile(
    r"securities\s+registered\s+pursuant\s+to\s+section\s*12\s*\(\s*g\s*\)"
    r"|indicate\s+by\s+check\s+mark"
    r"|emerging\s+growth\s+company"
    r"|the\s+registrant\s+had"
    r"|number\s+of\s+shares\s+of\s+common\s+stock\s+outstanding"
    r"|as\s+of\s+.+\s+there\s+were\s+.+\s+shares",
    re.IGNORECASE,
)
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
_COMMON_STOCK_RE = re.compile(r"\bcommon\s+stock\b|\bordinary\s+shares?\b", re.IGNORECASE)
_SYMBOL_RE = re.compile(r"^(?:[A-Z0-9][A-Z0-9.\-\/]{0,20}|—|–|--|-)$")
_EXCHANGE_RE = re.compile(
    r"\b(?:"
    r"New\s+York\s+Stock\s+Exchange(?:,\s*Inc\.)?"
    r"|Chicago\s+Stock\s+Exchange(?:,\s*Inc\.)?"
    r"|The\s+Nasdaq\s+Stock\s+Market\s+LLC"
    r"|Nasdaq(?:\s+Global\s+Select\s+Market|\s+Global\s+Market|\s+Capital\s+Market|\s+National\s+Market)?"
    r"|NASDAQ(?:\s+Global\s+Select\s+Market|\s+Global\s+Market|\s+Capital\s+Market|\s+National\s+Market)?"
    r"|NYSE(?:\s+American|\s+Arca)?(?:\s+LLC)?"
    r"|Stock\s+Exchange(?:\s+LLC)?"
    r"|Stock\s+Market(?:\s+LLC)?"
    r"|SIX\s+Swiss\s+Exchange"
    r"|Tel\s+Aviv\s+Stock\s+Exchange"
    r")\b",
    re.IGNORECASE,
)


def rule_section_12b_trading_table(doc: dict) -> list[dict]:
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
                or bool(_SECTION_RE.search(cleaned))
            )

        def is_symbol(text: str) -> bool:
            cleaned = norm(text)
            if not cleaned or any(ch.islower() for ch in cleaned):
                return False
            if _EXCHANGE_RE.search(cleaned):
                return False
            return bool(_SYMBOL_RE.match(cleaned))

        def is_exchange(text: str) -> bool:
            return bool(_EXCHANGE_RE.search(norm(text)))

        def is_probable_title(text: str) -> bool:
            cleaned = norm(text)
            if not cleaned:
                return False
            if re.fullmatch(r"\d{4,}", cleaned):
                return False
            if is_exchange(cleaned) or is_header(cleaned) or is_symbol(cleaned):
                return False
            return True

        def has_header_cluster(window: list[dict]) -> bool:
            title = False
            exchange = False
            symbol = False
            for item in window:
                cleaned = norm(item.get("text") or "")
                if not cleaned:
                    continue
                title = title or bool(_TITLE_HEADER_RE.search(cleaned))
                exchange = exchange or bool(_EXCHANGE_HEADER_RE.search(cleaned))
                symbol = symbol or bool(_SYMBOL_HEADER_RE.search(cleaned))
            return title and exchange and (symbol or len(window) >= 2)

        def make_span(texts: list[str], source: dict | None) -> list[dict]:
            if not texts:
                return []
            span = {"text": "; ".join(texts)}
            if source:
                for key in ("page_no", "line_no", "paragraph_no"):
                    if source.get(key) is not None:
                        span[key] = source.get(key)
            return [span]

        def add_row(rows: list[str], seen: set[str], title: str, symbol: str, exchange: str) -> None:
            title = norm(title)
            symbol = norm(symbol)
            exchange = norm(exchange)
            if not exchange:
                return
            if symbol:
                row = f"{title} | {symbol} | {exchange}" if title else f"{symbol} — {exchange}"
            elif title and _COMMON_STOCK_RE.search(title):
                row = f"{title} | {exchange}"
            else:
                return
            key = row.lower()
            if key not in seen:
                seen.add(key)
                rows.append(row)

        def extract_from_items(items: list[dict], require_section: bool) -> list[dict]:
            non_empty = []
            for item in items:
                if isinstance(item, dict) and norm(item.get("text") or ""):
                    non_empty.append(item)

            if not non_empty:
                return []

            start_idx = None
            if require_section:
                for idx, item in enumerate(non_empty):
                    if _SECTION_RE.search(norm(item.get("text") or "")):
                        start_idx = idx
                        break
                if start_idx is None:
                    return []
            else:
                max_scan = min(len(non_empty), 140)
                for idx in range(max_scan):
                    window = non_empty[idx : min(max_scan, idx + 6)]
                    if has_header_cluster(window):
                        header_positions = [idx + offset for offset, item in enumerate(window) if is_header(item.get("text") or "")]
                        start_idx = min(header_positions) if header_positions else idx
                        break
                if start_idx is None:
                    return []

            block: list[dict] = []
            header_mode = False
            for item in non_empty[start_idx:]:
                cleaned = norm(item.get("text") or "")
                if not cleaned:
                    continue
                if _END_RE.search(cleaned):
                    break
                if not header_mode:
                    header_mode = is_header(cleaned) or has_header_cluster(non_empty[start_idx : min(len(non_empty), start_idx + 6)])
                if is_header(cleaned):
                    continue
                block.append(item)

            if not block:
                return []

            rows: list[str] = []
            seen: set[str] = set()
            first_source: dict | None = None

            idx = 0
            while idx < len(block):
                title = norm(block[idx].get("text") or "")
                if not is_probable_title(title):
                    idx += 1
                    continue

                symbol = ""
                exchange = ""

                symbol_idx = None
                for j in range(idx + 1, min(len(block), idx + 5)):
                    candidate = norm(block[j].get("text") or "")
                    if not candidate:
                        continue
                    if is_symbol(candidate):
                        symbol = candidate
                        symbol_idx = j
                        break

                if symbol_idx is not None:
                    for j in range(symbol_idx + 1, min(len(block), symbol_idx + 4)):
                        candidate = norm(block[j].get("text") or "")
                        if is_exchange(candidate):
                            exchange = candidate
                            break
                else:
                    for j in range(idx + 1, min(len(block), idx + 3)):
                        candidate = norm(block[j].get("text") or "")
                        if is_exchange(candidate):
                            exchange = candidate
                            break

                if exchange:
                    if first_source is None:
                        first_source = block[symbol_idx] if symbol_idx is not None else block[idx]
                    add_row(rows, seen, title, symbol, exchange)
                    next_idx = (symbol_idx + 2) if symbol_idx is not None else (idx + 2)
                    if symbol_idx is not None:
                        while next_idx + 1 < len(block):
                            next_symbol = norm(block[next_idx].get("text") or "")
                            next_exchange = norm(block[next_idx + 1].get("text") or "")
                            if not (is_symbol(next_symbol) and is_exchange(next_exchange)):
                                break
                            add_row(rows, seen, title, next_symbol, next_exchange)
                            next_idx += 2
                    idx = next_idx
                    continue

                idx += 1

            if rows:
                if len(rows) == 1 and _COMMON_STOCK_RE.search(rows[0]):
                    rows[0] = rows[0].replace(" | ", ", ")
                return make_span(rows, first_source)

            titles = [norm(item.get("text") or "") for item in block if not is_exchange(item.get("text") or "")]
            exchanges = [norm(item.get("text") or "") for item in block if is_exchange(item.get("text") or "")]
            if titles and exchanges and len(titles) >= 1 and len(exchanges) >= 1:
                common_titles = [title for title in titles if _COMMON_STOCK_RE.search(title)]
                if common_titles:
                    add_row(rows, seen, common_titles[0], "", exchanges[0])
                else:
                    add_row(rows, seen, titles[0], "", exchanges[0])
                if rows:
                    if len(rows) == 1 and _COMMON_STOCK_RE.search(rows[0]):
                        rows[0] = rows[0].replace(" | ", ", ")
                    return make_span(rows, block[0])

            return []

        line_items = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        spans = extract_from_items(line_items, require_section=True)
        if spans:
            return spans

        spans = extract_from_items(line_items, require_section=False)
        if spans:
            return spans

        paragraph_items = [item for item in (doc.get("paragraphs") or []) if isinstance(item, dict)]
        spans = extract_from_items(paragraph_items, require_section=True)
        if spans:
            return spans

        spans = extract_from_items(paragraph_items, require_section=False)
        if spans:
            return spans

        return []
    except Exception:
        return []
