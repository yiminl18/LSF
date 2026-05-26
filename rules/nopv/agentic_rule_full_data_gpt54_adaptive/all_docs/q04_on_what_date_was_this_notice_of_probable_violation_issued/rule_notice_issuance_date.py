import re


_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_MONTH_RE = "|".join(_MONTHS)
_DATE_LINE_RE = re.compile(
    rf"^\s*({_MONTH_RE})\s+(\d{{1,2}})\s*,?\s*(\d{{4}})\s*$",
    re.IGNORECASE,
)
_DOC_NAME_DATE_RE = re.compile(r"_\s*(\d{8})(?:_|$)")


def rule_notice_issuance_date(doc: dict) -> list[dict]:
    try:
        def canonicalize(month: str, day: str, year: str) -> str:
            month_name = month[:1].upper() + month[1:].lower()
            return f"{month_name} {int(day)}, {year}"

        def match_line(text: str) -> str | None:
            normalized = " ".join((text or "").split())
            match = _DATE_LINE_RE.match(normalized)
            if not match:
                return None
            return canonicalize(match.group(1), match.group(2), match.group(3))

        def build_span(text: str, item: dict) -> dict:
            span = {"text": text}
            for key in ("page_no", "line_no", "paragraph_no"):
                if key in item:
                    span[key] = item[key]
            return span

        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        for item in lines[:50]:
            matched = match_line(item.get("text") or "")
            if matched:
                return [build_span(matched, item)]

        paragraphs = [item for item in (doc.get("paragraphs") or []) if isinstance(item, dict)]
        for item in paragraphs[:12]:
            for raw_line in (item.get("text") or "").splitlines():
                matched = match_line(raw_line)
                if matched:
                    return [build_span(matched, item)]

        match = _DOC_NAME_DATE_RE.search(doc.get("doc_name") or "")
        if match:
            raw = match.group(1)
            month = int(raw[:2])
            day = int(raw[2:4])
            year = raw[4:]
            if 1 <= month <= 12 and 1 <= day <= 31:
                return [{"text": f"{_MONTHS[month - 1]} {day}, {year}"}]

        return []
    except Exception:
        return []
