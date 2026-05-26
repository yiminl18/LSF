import re
from datetime import datetime


MONTH_PATTERN = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
DATE_PATTERN = rf"{MONTH_PATTERN}\.?\s+\d{{1,2}}(?:,)?\s+\d{{4}}"
DATE_LINE_RE = re.compile(rf"^{DATE_PATTERN}$", re.IGNORECASE)
FILED_LINE_RE = re.compile(rf"^Filed\s+({DATE_PATTERN})$", re.IGNORECASE)
AMENDED_LINE_RE = re.compile(rf"^Amended\s+({DATE_PATTERN})$", re.IGNORECASE)
OPINION_FILED_RE = re.compile(
    rf"\b[Tt]he\s+[Oo]pinion\s+filed\s+({DATE_PATTERN})\b",
    re.IGNORECASE,
)


def _normalize(text):
    return " ".join(str(text or "").split())


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


def _span(line, text):
    result = {"text": text}
    if line.get("page_no") is not None:
        result["page_no"] = line.get("page_no")
    if line.get("line_no") is not None:
        result["line_no"] = line.get("line_no")
    if line.get("paragraph_no") is not None:
        result["paragraph_no"] = line.get("paragraph_no")
    return result


def _header_lines(lines, max_pages=20, max_lines=2000):
    selected = []
    for line in lines:
        page_no = line.get("page_no")
        if isinstance(page_no, int) and page_no > max_pages:
            break
        selected.append(line)
        if len(selected) >= max_lines:
            break
    return selected


def _nonempty_window(lines, start, stop):
    values = []
    for line in lines[start:stop]:
        text = _normalize(line.get("text"))
        if text:
            values.append(text)
    return values


def _looks_like_header_context(prev_window, next_window):
    prev_hit = any(
        "Argued and Submitted" in text
        or re.search(r"(?i)^Submitted\b", text)
        or re.search(r"(?i)^(?:Appeal from|On Petition for Review)", text)
        or re.search(
            r",\s*(?:California|Washington|Oregon|Nevada|Arizona|Idaho|Montana|"
            r"Alaska|Hawaii|Guam|NMI|Saipan|Seattle|Pasadena|San Francisco|"
            r"Portland|Las Vegas|Honolulu)$",
            text,
        )
        for text in prev_window
    )
    next_hit = any(
        text.startswith("Before:")
        or text.startswith("Opinion by")
        or text.startswith("ORDER")
        or text.startswith("SUMMARY")
        for text in next_window
    )
    return prev_hit and next_hit


def _doc_name_fallback(doc_name):
    m = re.match(r"^(\d{4})(\d{2})(\d{2})_", str(doc_name or ""))
    if not m:
        return None
    try:
        dt = datetime.strptime("".join(m.groups()), "%Y%m%d")
    except ValueError:
        return None
    return f"{dt.strftime('%B')} {dt.day}, {dt.year}"


def rule_filed_date_primary(doc: dict) -> list[dict]:
    try:
        lines = _sorted_lines(doc)
        if not lines:
            return []

        header = _header_lines(lines)
        if not header:
            return []

        # Orders withdrawing or amending an earlier opinion often ask about the
        # underlying opinion's original filing date, which appears verbatim.
        for line in header:
            text = _normalize(line.get("text"))
            m = OPINION_FILED_RE.search(text)
            if m:
                return [_span(line, m.group(1))]

        explicit_filed = []
        for line in header:
            text = _normalize(line.get("text"))
            m = FILED_LINE_RE.match(text)
            if m:
                explicit_filed.append((line, m.group(1)))
        if explicit_filed:
            return [_span(explicit_filed[0][0], explicit_filed[0][1])]

        for idx, line in enumerate(header):
            text = _normalize(line.get("text"))
            if text.upper() != "FILED":
                continue
            for follow in header[idx + 1 : idx + 14]:
                candidate = _normalize(follow.get("text"))
                if not candidate or AMENDED_LINE_RE.match(candidate):
                    continue
                if DATE_LINE_RE.match(candidate):
                    return [_span(follow, candidate)]

        for idx, line in enumerate(header):
            text = _normalize(line.get("text"))
            if not DATE_LINE_RE.match(text):
                continue
            prev_window = _nonempty_window(header, max(0, idx - 6), idx)
            next_window = _nonempty_window(header, idx + 1, idx + 7)
            if _looks_like_header_context(prev_window, next_window):
                return [_span(line, text)]

        # A few long-caption orders do not expose the filing date in OCR'd text.
        full_text = doc.get("text") or ""
        if "AMENDED" not in full_text.upper() and "THE OPINION FILED" not in full_text.upper():
            fallback = _doc_name_fallback(doc.get("doc_name"))
            if fallback:
                return [{"text": fallback}]

        return []
    except Exception:
        return []
