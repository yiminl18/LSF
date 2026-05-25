import re


_HEADING_RE = re.compile(r"\bPROPOSED\s+COMPLIANCE\s+ORDER\b", re.IGNORECASE)
_STOP_RE = re.compile(
    r"^(it\s+is\s+requested(?:\s*\(not\s+mandated\))?|response\s+options|response\s+to\s+this\s+notice|"
    r"enclosures?:|sincerely|respectfully|cc\b|attachments?)",
    re.IGNORECASE,
)
_PAGE_RE = re.compile(r"^(page\s+\d+(?:\s+of\s+\d+)?)$", re.IGNORECASE)
_LABEL_RE = re.compile(r"^\s*(?:[A-Z]|\d+)\.(?:\s+|$)")
_FINAL_ORDER_RE = re.compile(r"\bfinal\s+order\b", re.IGNORECASE)
_TIME_RE = re.compile(
    r"\b(?:\d{1,4}\s+days?|one\s+year|a\s+year|12\s+months?|18\s+months?|24\s+months?)\b",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"\b(must|shall|provide|submit|complete|correct|repair|install|update|develop|implement|"
    r"conduct|inspect|demonstrate|replace|remove|revise|restore|calibrate|survey|test|document)\b",
    re.IGNORECASE,
)
_DAYS_RE = re.compile(r"\b(\d{1,4})\s+days?\b", re.IGNORECASE)
_MONTHS_RE = re.compile(r"\b(\d{1,2})\s+months?\b", re.IGNORECASE)


def rule_proposed_compliance_order_deadlines(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        if not lines:
            return []

        heading_idxs = [
            i
            for i, item in enumerate(lines)
            if _HEADING_RE.search(norm(item.get("text") or ""))
        ]
        if not heading_idxs:
            return []

        # The notice body usually contains an earlier heading; the attached order is the last one.
        start_idx = heading_idxs[-1] + 1

        def is_noise(text: str) -> bool:
            text = norm(text)
            return not text or _PAGE_RE.fullmatch(text) is not None

        chunks: list[list[dict]] = []
        current: list[dict] = []
        for item in lines[start_idx:]:
            text = norm(item.get("text") or "")
            if _STOP_RE.match(text):
                break
            if is_noise(text):
                if current:
                    chunks.append(current)
                    current = []
                continue
            if text == "\f":
                if current:
                    chunks.append(current)
                    current = []
                continue
            if current and _LABEL_RE.match(text):
                chunks.append(current)
                current = [item]
                continue
            current.append(item)

        if current:
            chunks.append(current)

        best_span: dict | None = None
        best_days = -1
        for chunk in chunks:
            chunk_text = "\n".join(norm(item.get("text") or "") for item in chunk).strip()
            if not chunk_text:
                continue
            low = chunk_text.lower()
            if low.startswith("it is requested"):
                continue
            if not _FINAL_ORDER_RE.search(low):
                continue
            if not _TIME_RE.search(low):
                continue
            if not _ACTION_RE.search(low):
                continue

            days = 0
            day_values = [int(match.group(1)) for match in _DAYS_RE.finditer(low)]
            month_values = [int(match.group(1)) * 30 for match in _MONTHS_RE.finditer(low)]
            if re.search(r"\b(one\s+year|a\s+year)\b", low):
                day_values.append(365)
            if re.search(r"\b12\s+months?\b", low):
                month_values.append(365)
            if re.search(r"\b18\s+months?\b", low):
                month_values.append(540)
            if re.search(r"\b24\s+months?\b", low):
                month_values.append(730)
            if day_values or month_values:
                days = max(day_values + month_values)
            else:
                continue

            span = {"text": chunk_text}
            first = chunk[0]
            for key in ("page_no", "line_no"):
                if key in first:
                    span[key] = first[key]
            if days > best_days:
                best_days = days
                best_span = span

        return [best_span] if best_span is not None else []
    except Exception:
        return []
