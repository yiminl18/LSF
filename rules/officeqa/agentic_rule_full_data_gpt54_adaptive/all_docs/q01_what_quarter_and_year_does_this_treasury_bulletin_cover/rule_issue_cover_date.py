import re


MONTH_YEAR_RE = re.compile(
    r"\b(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
SEASON_RE = re.compile(r"\b(?P<season>Winter|Spring|Summer|Fall|Autumn)\s+Issue\b", re.IGNORECASE)
QUARTER_RE = re.compile(
    r"\b(?P<quarter>First|Second|Third|Fourth|1st|2nd|3rd|4th)[- ]Quarter\b",
    re.IGNORECASE,
)
FISCAL_YEAR_RE = re.compile(r"\bFiscal\b[^0-9]{0,12}(?P<year>(?:19|20)\s*\d{2}|\d\s*\d\s*\d\s*\d)", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(?P<year>(?:19|20)\d{2})\b")
DOC_NAME_RE = re.compile(r"treasury_bulletin_(?P<year>\d{4})_(?P<month>\d{2})", re.IGNORECASE)
SEASON_TO_QUARTER = {
    "winter": "First Quarter",
    "spring": "Second Quarter",
    "summer": "Third Quarter",
    "fall": "Fourth Quarter",
    "autumn": "Fourth Quarter",
}


def rule_issue_cover_date(doc: dict) -> list[dict]:
    try:
        spans = []
        seen = set()
        lines = doc.get("lines") or []
        doc_name = doc.get("doc_name") or ""
        doc_match = DOC_NAME_RE.search(doc_name)
        doc_year = doc_match.group("year") if doc_match else None
        doc_month_num = int(doc_match.group("month")) if doc_match else None
        month_names = [
            "",
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
        ]
        doc_month_name = month_names[doc_month_num] if doc_month_num and 1 <= doc_month_num <= 12 else None
        early_lines = lines[:250]

        for idx, line in enumerate(early_lines):
            text = " ".join(((line.get("text") or "").strip()).split())
            if not text:
                continue

            def add_span(span_text: str, source_line: dict | None = None) -> None:
                cleaned = " ".join((span_text or "").split())
                if not cleaned:
                    return
                key = (
                    cleaned.lower(),
                    source_line.get("page_no") if source_line else None,
                    source_line.get("line_no") if source_line else None,
                )
                if key in seen:
                    return
                seen.add(key)
                span = {"text": cleaned}
                if source_line is not None and source_line.get("page_no") is not None:
                    span["page_no"] = source_line["page_no"]
                if source_line is not None and source_line.get("line_no") is not None:
                    span["line_no"] = source_line["line_no"]
                spans.append(span)

            season_match = SEASON_RE.search(text)
            if season_match:
                season = season_match.group("season").lower()
                nearby_text = " ".join(
                    " ".join(((early_lines[j].get("text") or "").strip()).split())
                    for j in range(max(0, idx - 2), min(len(early_lines), idx + 6))
                    if (early_lines[j].get("text") or "").strip()
                )
                fiscal_match = FISCAL_YEAR_RE.search(nearby_text)
                year = fiscal_match.group("year").replace(" ", "") if fiscal_match else doc_year
                quarter_name = SEASON_TO_QUARTER.get(season)
                if year and quarter_name:
                    add_span(f"{quarter_name}, Fiscal {year}", line)
                    add_span(text, line)
                    if doc_month_name:
                        add_span(f"{doc_month_name} {doc_year}", line)
                    return spans

            month_match = MONTH_YEAR_RE.search(text)
            if month_match:
                for look_near in range(max(0, idx - 10), min(len(early_lines), idx + 16)):
                    nearby = " ".join(((early_lines[look_near].get("text") or "").strip()).split())
                    if "treasury bulletin" in nearby.lower():
                        add_span(f"Treasury Bulletin {month_match.group(0)}", early_lines[look_near])
                        break
                add_span(month_match.group(0), line)
                return spans

            if QUARTER_RE.search(text) and any(
                cue in text.lower() for cue in ("analysis", "budget results", "receipts by source", "fiscal", "issue")
            ):
                quarter_match = QUARTER_RE.search(text)
                quarter_name = quarter_match.group("quarter").replace("-", " ").title() + " Quarter"
                year = None
                fiscal_year = None
                fiscal_match = FISCAL_YEAR_RE.search(text)
                if fiscal_match:
                    fiscal_year = fiscal_match.group("year").replace(" ", "")
                    year = fiscal_year
                if year is None:
                    for look_idx in range(max(0, idx - 2), min(len(early_lines), idx + 3)):
                        nearby = " ".join(((early_lines[look_idx].get("text") or "").strip()).split())
                        if not nearby:
                            continue
                        fiscal_match = FISCAL_YEAR_RE.search(nearby)
                        if fiscal_match:
                            fiscal_year = fiscal_match.group("year").replace(" ", "")
                            year = fiscal_year
                            break
                        year_match = YEAR_RE.search(nearby)
                        if year_match:
                            year = year_match.group("year")
                            break
                if year is None:
                    year = doc_year
                if year:
                    if fiscal_year or "fiscal" in text.lower():
                        add_span(f"{quarter_name}, Fiscal {year}", line)
                    else:
                        add_span(f"{quarter_name} {year}", line)
                add_span(text, line)
                if doc_month_name and doc_year:
                    add_span(f"{doc_month_name} {doc_year}", line)
                return spans

        if doc_month_name and doc_year:
            spans.append({"text": f"{doc_month_name} {doc_year}"})

        return spans
    except Exception:
        return []
