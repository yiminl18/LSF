import re


AVG_PATTERNS = [
    re.compile(
        r"\b(?:increases in jobs on nonfarm payrolls(?: have)? slowed to an average of|increases in jobs on nonfarm payrolls averaged|nonfarm payroll employment(?: rose| increased| grew)?(?: an)? average of|nonfarm payroll employment averaged|growth of nonfarm payroll employment averaged|gains in nonfarm payrolls employment averaged|job gains through [A-Za-z]+ averaged|job gains in [A-Za-z]+ averaged|nonfarm payroll employment rose an average of)\s+[\d,]+(?:\.\d+)?\s+(?:per month|a month)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:average for the entire year so far of|average for the year so far of|average for the entire year so far is|average for the year so far is)\s+[\d,]+(?:\.\d+)?\s+(?:per month|a month)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:average(?:d)? monthly (?:growth|increase) of|monthly (?:growth|increase) of)\s+[\d,]+(?:\.\d+)?\s+(?:per month|a month)\b",
        re.IGNORECASE,
    ),
]


def _normalize(text: str) -> str:
    text = text or ""
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _sentences(text: str) -> list[str]:
    text = _normalize(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def _matches(text: str) -> bool:
    text = _normalize(text)
    if not text:
        return False
    low = text.lower()
    if "payroll" not in low and "nonfarm" not in low and "job" not in low:
        return False
    return any(pattern.search(text) for pattern in AVG_PATTERNS)


def _extract_fragment(text: str) -> str | None:
    text = _normalize(text)
    if not text:
        return None
    low = text.lower()
    if "payroll" not in low and "nonfarm" not in low and "job" not in low:
        return None
    for pattern in AVG_PATTERNS:
        m = pattern.search(text)
        if m:
            return _normalize(m.group(0))
    return None


def rule_nonfarm_payroll_average(doc: dict) -> list[dict]:
    try:
        seen = set()
        spans = []

        for para in doc.get("paragraphs") or []:
            text = para.get("text") or ""
            if not text:
                continue
            for sent in _sentences(text):
                fragment = _extract_fragment(sent)
                if not fragment:
                    continue
                key = (para.get("page_no"), para.get("paragraph_no"), fragment)
                if key in seen:
                    continue
                seen.add(key)
                span = {"text": fragment}
                if para.get("page_no") is not None:
                    span["page_no"] = para["page_no"]
                if para.get("paragraph_no") is not None:
                    span["paragraph_no"] = para["paragraph_no"]
                spans.append(span)

        if spans:
            return spans

        lines = doc.get("lines") or []
        for idx, line in enumerate(lines):
            if not _extract_fragment(line.get("text") or ""):
                continue
            window_parts = []
            for j in range(max(0, idx - 2), min(len(lines), idx + 4)):
                part = _normalize(lines[j].get("text") or "")
                if part:
                    window_parts.append(part)
            window = _normalize(" ".join(window_parts))
            fragment = _extract_fragment(window)
            if not fragment:
                continue
            key = (line.get("page_no"), line.get("line_no"), fragment)
            if key in seen:
                continue
            seen.add(key)
            span = {"text": fragment}
            if line.get("page_no") is not None:
                span["page_no"] = line["page_no"]
            if line.get("line_no") is not None:
                span["line_no"] = line["line_no"]
            spans.append(span)

        return spans
    except Exception:
        return []
