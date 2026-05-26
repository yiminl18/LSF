import re


_HEADING_RE = re.compile(r"^\s*\ufeff*\f*PROPOSED\s+COMPLIANCE\s+ORDER\s*$", re.IGNORECASE)
_STOP_RE = re.compile(
    r"^(it\s+is\s+requested(?:\s*\(not\s+mandated\))?|response\s+options|response\s+to\s+this\s+notice|"
    r"enclosures?:|sincerely|respectfully|cc\b|attachments?)",
    re.IGNORECASE,
)
_PAGE_RE = re.compile(r"^(page\s+\d+(?:\s+of\s+\d+)?)$", re.IGNORECASE)
_LABEL_RE = re.compile(r"^\s*(?:[A-Z]|\d+)\.(?:\s+|$)")
_FINAL_ORDER_RE = re.compile(r"\bfinal\s+order\b", re.IGNORECASE)
_TIME_RE = re.compile(
    r"\b(?:\d{1,4}\s+days?|\d{1,2}\s+months?|\d{1,2}\s+years?|one\s+year|a\s+year|12\s+months?|18\s+months?|24\s+months?)\b",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"\b(must|shall|provide|submit|complete|correct|repair|install|update|develop|implement|"
    r"conduct|inspect|demonstrate|replace|remove|revise|restore|calibrate|survey|test|document)\b",
    re.IGNORECASE,
)
_PRIMARY_ACTION_RE = re.compile(
    r"\b(complete|correct|repair|install|update|develop|implement|conduct|inspect|demonstrate|"
    r"replace|remove|restore|calibrate|survey|test|revise|maintain|train|perform|excavate|recoat|secure|close|verify)\b",
    re.IGNORECASE,
)
_SECONDARY_RE = re.compile(
    r"\b(documentation|records?|evidence|proof)\b",
    re.IGNORECASE,
)
_REPORT_ONLY_RE = re.compile(
    r"\b(report(?:s|ing)?|progress|follow[- ]?up|outstanding work necessary to implement|"
    r"days thereafter|until all work necessary to implement|91st day)\b",
    re.IGNORECASE,
)
_DAYS_RE = re.compile(r"\b(\d{1,4})\s+days?\b", re.IGNORECASE)
_MONTHS_RE = re.compile(r"\b(\d{1,2})\s+months?\b", re.IGNORECASE)
_YEARS_RE = re.compile(r"\b(\d{1,2})\s+years?\b", re.IGNORECASE)


def rule_proposed_compliance_order_deadlines(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "").replace("\ufeff", "").replace("\f", "")).strip()

        def materialize(items: list[dict], text_key: str) -> list[dict]:
            return [
                {
                    "text": item.get(text_key) or "",
                    "page_no": item.get("page_no"),
                    "line_no": item.get("line_no"),
                    "paragraph_no": item.get("paragraph_no"),
                }
                for item in items
                if isinstance(item, dict)
            ]

        lines = materialize(doc.get("lines") or [], "text")
        if not lines:
            lines = materialize(doc.get("paragraphs") or [], "text")
        if not lines:
            raw_text = doc.get("text") or ""
            if not raw_text:
                return []
            lines = [{"text": raw_text, "page_no": None, "line_no": None, "paragraph_no": None}]

        heading_indices = [
            i for i, item in enumerate(lines) if _HEADING_RE.search(norm(item.get("text") or ""))
        ]
        if not heading_indices:
            return []
        heading_idx = heading_indices[-1]

        best_span: dict | None = None
        best_days = -1

        for i in range(heading_idx + 1, len(lines)):
            item = lines[i]
            text = norm(item.get("text") or "")
            if _STOP_RE.match(text):
                break
            if not text or _PAGE_RE.fullmatch(text) is not None:
                continue

            neighborhood_items = lines[max(heading_idx + 1, i - 1) : min(len(lines), i + 2)]
            neighborhood_text = " ".join(
                norm(entry.get("text") or "")
                for entry in neighborhood_items
                if norm(entry.get("text") or "")
            ).lower()
            if not neighborhood_text:
                continue
            if not _FINAL_ORDER_RE.search(neighborhood_text):
                continue
            if not _ACTION_RE.search(neighborhood_text):
                continue

            for sentence in re.split(r"(?<=[.!?])\s+", neighborhood_text):
                sentence = sentence.strip()
                if not sentence or not _TIME_RE.search(sentence):
                    continue

                for match in _TIME_RE.finditer(sentence):
                    raw = match.group(0)
                    days = None
                    day_match = re.fullmatch(r"(\d{1,4})\s+days?", raw, re.IGNORECASE)
                    month_match = re.fullmatch(r"(\d{1,2})\s+months?", raw, re.IGNORECASE)
                    year_match = re.fullmatch(r"(\d{1,2})\s+years?", raw, re.IGNORECASE)
                    if day_match:
                        days = int(day_match.group(1))
                    elif month_match:
                        days = int(month_match.group(1)) * 30
                    elif year_match:
                        days = int(year_match.group(1)) * 365
                    elif re.fullmatch(r"(one\s+year|a\s+year)", raw, re.IGNORECASE):
                        days = 365
                    elif re.fullmatch(r"12\s+months?", raw, re.IGNORECASE):
                        days = 365
                    elif re.fullmatch(r"18\s+months?", raw, re.IGNORECASE):
                        days = 540
                    elif re.fullmatch(r"24\s+months?", raw, re.IGNORECASE):
                        days = 730
                    if days is None:
                        continue

                    if _REPORT_ONLY_RE.search(sentence):
                        continue

                    span = {
                        "text": "\n".join(
                            norm(entry.get("text") or "")
                            for entry in neighborhood_items
                            if norm(entry.get("text") or "")
                        )
                    }
                    first = neighborhood_items[0]
                    for key in ("page_no", "line_no", "paragraph_no"):
                        value = first.get(key)
                        if value is not None:
                            span[key] = value

                    if days > best_days:
                        best_days = days
                        best_span = span

        return [best_span] if best_span is not None else []
    except Exception:
        return []
