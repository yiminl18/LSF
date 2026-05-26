import re


MONTH_PATTERN = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?\.?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
DATE_RE = re.compile(rf"^{MONTH_PATTERN}\s+\d{{1,2}}(?:,)?\s+\d{{4}}$", re.IGNORECASE)


def _normalize(text):
    return " ".join(str(text or "").split())


def _span(line, text):
    span = {"text": text}
    if line.get("page_no") is not None:
        span["page_no"] = line.get("page_no")
    if line.get("line_no") is not None:
        span["line_no"] = line.get("line_no")
    if line.get("paragraph_no") is not None:
        span["paragraph_no"] = line.get("paragraph_no")
    return span


def _sorted_lines(doc):
    lines = doc.get("lines") or []
    keyed = []
    for idx, line in enumerate(lines):
        keyed.append(
            (
                -1 if line.get("page_no") is None else line.get("page_no"),
                idx if line.get("line_no") is None else line.get("line_no"),
                idx,
                line,
            )
        )
    keyed.sort()
    return [line for _, _, _, line in keyed]


def rule_header_standalone_date(doc: dict) -> list[dict]:
    try:
        lines = _sorted_lines(doc)
        if not lines:
            return []

        if any(
            re.match(r"(?i)^filed\s+.+$", _normalize(line.get("text"))) or _normalize(line.get("text")).upper() == "FILED"
            for line in lines
        ):
            return []

        # Handle headers where the filing date is a bare date line between the
        # argument/location block and the judge panel.
        for idx, line in enumerate(lines):
            text = _normalize(line.get("text"))
            if not DATE_RE.match(text):
                continue
            prev_window = [_normalize(prev.get("text")) for prev in lines[max(0, idx - 3) : idx]]
            next_window = [_normalize(nxt.get("text")) for nxt in lines[idx + 1 : idx + 4]]
            prev_hit = any(
                "Argued and Submitted" in t
                or t.endswith(", California")
                or t.endswith(", Washington")
                or t.endswith(", Nevada")
                or t.endswith(", Oregon")
                or t.endswith(", Arizona")
                or t.endswith(", Idaho")
                or t.endswith(", Alaska")
                for t in prev_window
            )
            next_hit = any(t.startswith("Before:") or t.startswith("Opinion by") or t.startswith("SUMMARY") for t in next_window)
            if prev_hit and next_hit:
                return [_span(line, text)]

        return []
    except Exception:
        return []
