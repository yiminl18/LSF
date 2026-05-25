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
_HEADER_DATE_RE = re.compile(
    rf"^\s*({_MONTH_RE})\s+(\d{{1,2}})\s*,?\s*(\d{{4}})\s*$",
    re.IGNORECASE,
)
_DOC_NAME_DATE_RE = re.compile(r"_(\d{8})(?:_|$)")


def rule_notice_issuance_date(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def canonicalize(month: str, day: str, year: str) -> str:
            month = month[:1].upper() + month[1:].lower()
            return f"{month} {int(day)}, {year}"

        def header_date_from_items(items: list[dict], limit: int = 35) -> dict | None:
            for item in items[:limit]:
                text = norm(item.get("text") or "")
                if not text:
                    continue
                match = _HEADER_DATE_RE.match(text)
                if match:
                    span = {"text": canonicalize(match.group(1), match.group(2), match.group(3))}
                    for key in ("page_no", "line_no", "paragraph_no"):
                        if key in item:
                            span[key] = item[key]
                    return span
            return None

        # Prefer the explicit header date from the top of the document.
        for source_key in ("lines", "paragraphs"):
            items = [s for s in (doc.get(source_key) or []) if isinstance(s, dict)]
            if not items:
                continue
            found = header_date_from_items(items)
            if found:
                return [found]

        pages = [s for s in (doc.get("pages") or []) if isinstance(s, dict)]
        if pages:
            found = header_date_from_items(pages, limit=2)
            if found:
                return [found]

        # Fallback: derive the issue date from the encoded document name.
        doc_name = doc.get("doc_name") or ""
        match = _DOC_NAME_DATE_RE.search(doc_name)
        if match:
            raw = match.group(1)
            month = int(raw[:2])
            day = int(raw[2:4])
            year = raw[4:]
            if 1 <= month <= 12 and 1 <= day <= 31:
                month_name = _MONTHS[month - 1]
                return [{"text": f"{month_name} {day}, {year}"}]

        return []
    except Exception:
        return []
