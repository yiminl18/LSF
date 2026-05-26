import re


MONTH_PATTERN = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?\.?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
DATE_BODY = rf"{MONTH_PATTERN}\s+\d{{1,2}}(?:,)?\s+\d{{4}}"
DATE_RE = re.compile(rf"^{DATE_BODY}$", re.IGNORECASE)
FILED_RE = re.compile(rf"^filed\s+({DATE_BODY})$", re.IGNORECASE)


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


def rule_filed_date(doc: dict) -> list[dict]:
    try:
        lines = _sorted_lines(doc)
        if not lines:
            return []

        # Prefer an explicit "Filed Month day, year" header line when present.
        for line in lines:
            text = _normalize(line.get("text"))
            m = FILED_RE.match(text)
            if m:
                return [_span(line, m.group(1))]

        # Otherwise, look for a FILED stamp and capture the date printed nearby.
        for idx, line in enumerate(lines):
            text = _normalize(line.get("text"))
            if text.upper() == "FILED":
                for follow in lines[idx + 1 : idx + 12]:
                    candidate = _normalize(follow.get("text"))
                    if DATE_RE.match(candidate):
                        return [_span(follow, candidate)]
            elif text.upper().startswith("FILED "):
                m = FILED_RE.match(text)
                if m:
                    return [_span(line, m.group(1))]

        return []
    except Exception:
        return []
