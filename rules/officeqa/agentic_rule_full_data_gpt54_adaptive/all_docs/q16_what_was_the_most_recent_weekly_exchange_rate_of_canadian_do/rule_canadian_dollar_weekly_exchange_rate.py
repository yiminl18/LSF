import re


_NEW_SECTION_START_RE = re.compile(
    r"SECTION\s+I[\.\-—~]*\s*Canadian Dollar Positions",
    re.IGNORECASE,
)
_NEW_TABLE_START_RE = re.compile(
    r"TABLE\s+FCP-I-1[\.\-—~]*.*Weekly Report of Major Market Participants",
    re.IGNORECASE,
)
_OLD_TABLE_START_RE = re.compile(
    r"Table\s+FCP-[Il1]{2}-2\.?\s*[-–—]?\s*Weekly\s+Bank\s+Positions",
    re.IGNORECASE,
)
_OLD_CANADIAN_HINT_RE = re.compile(
    r"Canadian\s+Dollar\s+Positions|Table\s+FCP-[Il1]{2}-1",
    re.IGNORECASE,
)
_CANADIAN_RATE_RE = re.compile(
    r"Canadian\s+dollars?\s+per\s+U\.S\.?\s+dollar",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"(?:\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}\s*-\s*[A-Za-z]{3,9}\b)",
    re.IGNORECASE,
)
_DATE_LINE_RE = re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{2,4}\.?\s*$")
_NUMERIC_RE = re.compile(r"^[+-]?\d[\d,]*(?:\.\d+)?$")
_NUMERIC_TOKEN_RE = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")
_RATE_TOKEN_RE = re.compile(r"\b(?:0|1|2)\.\d{3,4}\b")
_END_RE = re.compile(
    r"^(?:SECTION\s+(?!I\b)|TABLE\s+FCP-(?!I-1\b)|Foreign Exchange Rates\b|"
    r"FOREIGN EXCHANGE RATES\b|FOREIGN CURRENCY POSITIONS\b|EXCHANGE STABILIZATION FUND\b)",
    re.IGNORECASE,
)


def rule_canadian_dollar_weekly_exchange_rate(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def is_numeric(text: str) -> bool:
            cleaned = norm(text).replace("$", "")
            return bool(_NUMERIC_RE.match(cleaned) or cleaned.lower() in {"n.a.", "na"})

        def extract_row_text(row: list[dict]) -> str:
            parts = [norm(item.get("text") or "") for item in row if norm(item.get("text") or "")]
            return norm(" ".join(parts))

        def row_rate(row: list[dict]) -> str | None:
            row_text = extract_row_text(row)
            if not row_text:
                return None
            date_match = _DATE_RE.search(row_text)
            tail = row_text[date_match.end() :] if date_match else row_text
            tokens = [tok for tok in _NUMERIC_TOKEN_RE.findall(tail) if is_numeric(tok)]
            if not tokens:
                return None
            decimals = [tok for tok in tokens if "." in tok]
            return (decimals[-1] if decimals else tokens[-1]).strip()

        def build_span(row: list[dict]) -> dict | None:
            rate = row_rate(row)
            row_text = extract_row_text(row)
            if not rate or not row_text:
                return None
            date_match = _DATE_RE.search(row_text)
            if date_match:
                text = norm(f"{row_text[:date_match.end()]} {rate}")
            else:
                text = norm(f"{row_text} {rate}")
            first = row[0] if row else {}
            span = {"text": text}
            for key in ("page_no", "paragraph_no", "line_no"):
                if first.get(key) is not None:
                    span[key] = first.get(key)
            return span

        def parse_modern_block(items: list[dict]) -> list[dict]:
            rows: list[list[dict]] = []
            current: list[dict] = []
            saw_date = False

            for item in items:
                text = norm(item.get("text") or "")
                if not text:
                    continue
                if _END_RE.search(text) and rows:
                    break
                if _NEW_SECTION_START_RE.search(text) or _NEW_TABLE_START_RE.search(text) or _CANADIAN_RATE_RE.search(text):
                    continue
                if _DATE_RE.search(text):
                    if current and saw_date:
                        rows.append(current)
                    current = [item]
                    saw_date = True
                    continue
                if current and (is_numeric(text) or _NUMERIC_TOKEN_RE.search(text)):
                    current.append(item)
            if current and saw_date:
                rows.append(current)

            if not rows:
                return []
            span = build_span(rows[-1])
            return [span] if span else []

        def scope_from_lines(lines: list[dict], start_idx: int) -> list[dict]:
            block: list[dict] = []
            for item in lines[start_idx : min(len(lines), start_idx + 1200)]:
                text = norm(item.get("text") or "")
                if block and _END_RE.search(text):
                    break
                block.append(item)
            return block

        def parse_legacy_block(lines: list[dict]) -> list[dict]:
            spans: list[dict] = []
            for idx, item in enumerate(lines):
                text = norm(item.get("text") or "")
                if not _OLD_TABLE_START_RE.search(text):
                    continue

                lookback_start = max(0, idx - 180)
                window = lines[lookback_start : idx + 1]
                joined = " ".join(norm(entry.get("text") or "") for entry in window)
                if not _OLD_CANADIAN_HINT_RE.search(joined):
                    continue

                date_positions = [
                    pos for pos, entry in enumerate(window) if _DATE_LINE_RE.match(norm(entry.get("text") or ""))
                ]
                if len(date_positions) < 2:
                    continue

                rate_hits = [
                    (pos, match.group(0))
                    for pos, entry in enumerate(window)
                    for match in _RATE_TOKEN_RE.finditer(norm(entry.get("text") or ""))
                ]
                if not rate_hits:
                    continue

                last_date = norm(window[date_positions[-1]].get("text") or "").rstrip(".")
                last_rate = rate_hits[-1][1]
                source = window[date_positions[-1]]
                spans.append(
                    {
                        "text": norm(f"Canadian dollar weekly bank positions {last_date} {last_rate}"),
                        "page_no": source.get("page_no"),
                        "line_no": source.get("line_no"),
                    }
                )
                return spans
            return []

        paragraphs = [p for p in (doc.get("paragraphs") or []) if isinstance(p, dict)]
        for para in paragraphs:
            text = para.get("text") or ""
            ntext = norm(text)
            if not ntext:
                continue
            if not (
                _NEW_SECTION_START_RE.search(ntext)
                and _NEW_TABLE_START_RE.search(ntext)
                and _CANADIAN_RATE_RE.search(ntext)
            ):
                continue
            raw_lines = [line for line in text.splitlines() if norm(line)]
            if not any(_DATE_RE.search(norm(line)) for line in raw_lines):
                continue
            items = []
            for idx, line in enumerate(raw_lines, start=1):
                items.append(
                    {
                        "text": line,
                        "page_no": para.get("page_no"),
                        "paragraph_no": para.get("paragraph_no"),
                        "line_no": idx,
                    }
                )
            spans = parse_modern_block(items)
            if spans:
                return spans

        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        if not lines:
            return []

        candidate_starts: list[int] = []
        for idx, item in enumerate(lines):
            text = norm(item.get("text") or "")
            if not text or not _NEW_TABLE_START_RE.search(text):
                continue
            context = " ".join(norm(lines[j].get("text") or "") for j in range(max(0, idx - 6), idx + 2))
            if not _NEW_SECTION_START_RE.search(context) and not _CANADIAN_RATE_RE.search(context):
                continue
            candidate_starts.append(idx)

        for start_idx in candidate_starts:
            spans = parse_modern_block(scope_from_lines(lines, start_idx))
            if spans:
                return spans

        for idx, item in enumerate(lines):
            text = norm(item.get("text") or "")
            if not _NEW_SECTION_START_RE.search(text):
                continue
            context = " ".join(norm(lines[j].get("text") or "") for j in range(idx, min(len(lines), idx + 18)))
            if not _NEW_TABLE_START_RE.search(context):
                continue
            spans = parse_modern_block(scope_from_lines(lines, idx))
            if spans:
                return spans

        return parse_legacy_block(lines)
    except Exception:
        return []
