import re


_SECTION_START_RE = re.compile(
    r"SECTION\s+I[\.\-—~]*\s*Canadian Dollar Positions",
    re.IGNORECASE,
)
_TABLE_START_RE = re.compile(
    r"TABLE\s+FCP-I-1[\.\-—~]*.*Weekly Report of Major Market Participants",
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
_NUMERIC_RE = re.compile(r"^[+-]?\d[\d,]*(?:\.\d+)?$")
_NUMERIC_TOKEN_RE = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")
_END_RE = re.compile(
    r"^(?:SECTION\s+(?!I\b)|TABLE\s+FCP-(?!I-1\b)|Foreign Exchange Rates\b|FOREIGN EXCHANGE RATES\b|FOREIGN CURRENCY POSITIONS\b)",
    re.IGNORECASE,
)


def rule_canadian_dollar_weekly_exchange_rate(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def is_numeric(text: str) -> bool:
            t = norm(text).replace("$", "")
            return bool(_NUMERIC_RE.match(t) or t.lower() in {"n.a.", "na"})

        def extract_row_text(row: list[dict]) -> str:
            parts = [norm(item.get("text") or "") for item in row if norm(item.get("text") or "")]
            return norm(" ".join(parts))

        def row_rate(row: list[dict]) -> str | None:
            row_text = extract_row_text(row)
            if not row_text:
                return None
            date_match = _DATE_RE.search(row_text)
            if date_match:
                tail = row_text[date_match.end() :]
            else:
                tail = row_text
            tokens = [tok for tok in _NUMERIC_TOKEN_RE.findall(tail) if is_numeric(tok)]
            if not tokens:
                return None
            decimals = [tok for tok in tokens if "." in tok]
            return (decimals[-1] if decimals else tokens[-1]).strip()

        def build_span(row: list[dict]) -> dict | None:
            rate = row_rate(row)
            if not rate:
                return None
            row_text = extract_row_text(row)
            if not row_text:
                return None
            # Keep the date when present, but make the answer-bearing number easy to spot.
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

        def parse_block(items: list[dict]) -> list[dict]:
            rows: list[list[dict]] = []
            current: list[dict] = []
            saw_date = False

            for item in items:
                text = norm(item.get("text") or "")
                if not text:
                    continue
                if _END_RE.search(text) and rows:
                    break
                if _SECTION_START_RE.search(text) or _TABLE_START_RE.search(text) or _CANADIAN_RATE_RE.search(text):
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

        # Prefer paragraph/page scope when it cleanly isolates the Canadian table.
        paragraphs = [p for p in (doc.get("paragraphs") or []) if isinstance(p, dict)]
        for para in paragraphs:
            text = para.get("text") or ""
            ntext = norm(text)
            if not ntext:
                continue
            if not (
                _SECTION_START_RE.search(ntext)
                and _TABLE_START_RE.search(ntext)
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
            spans = parse_block(items)
            if spans:
                return spans

        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        if not lines:
            return []

        candidate_starts: list[int] = []
        for idx, item in enumerate(lines):
            text = norm(item.get("text") or "")
            if not text:
                continue
            if not _TABLE_START_RE.search(text):
                continue
            context = " ".join(norm(lines[j].get("text") or "") for j in range(max(0, idx - 6), idx + 2))
            if not _SECTION_START_RE.search(context) and not _CANADIAN_RATE_RE.search(context):
                continue
            candidate_starts.append(idx)

        for start_idx in candidate_starts:
            spans = parse_block(scope_from_lines(lines, start_idx))
            if spans:
                return spans

        # Final fallback: locate the Canadian section and scan until the next section.
        for idx, item in enumerate(lines):
            text = norm(item.get("text") or "")
            if not _SECTION_START_RE.search(text):
                continue
            context = " ".join(norm(lines[j].get("text") or "") for j in range(idx, min(len(lines), idx + 18)))
            if not _TABLE_START_RE.search(context):
                continue
            spans = parse_block(scope_from_lines(lines, idx))
            if spans:
                return spans

        return []
    except Exception:
        return []
