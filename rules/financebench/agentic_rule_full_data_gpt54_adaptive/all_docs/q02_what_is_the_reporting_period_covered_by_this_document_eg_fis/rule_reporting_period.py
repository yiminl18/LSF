import re
from datetime import datetime


def rule_reporting_period(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            text = (text or "").replace("\u00a0", " ")
            return re.sub(r"\s+", " ", text).strip()

        def make_span(text: str, src: dict | None = None) -> dict:
            span = {"text": norm(text)}
            if src:
                if src.get("page_no") is not None:
                    span["page_no"] = src.get("page_no")
                if src.get("line_no") is not None:
                    span["line_no"] = src.get("line_no")
                if src.get("paragraph_no") is not None:
                    span["paragraph_no"] = src.get("paragraph_no")
            return span

        def format_doc_date(value: str) -> str:
            try:
                dt = datetime.strptime(value, "%Y-%m-%d")
                return f"{dt.strftime('%B')} {dt.day}, {dt.year}"
            except Exception:
                return value

        month = (
            r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
            r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
            r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
        )
        date_re = rf"{month}\s+\d{{1,2}},\s+\d{{4}}"

        lines = sorted(
            doc.get("lines") or [],
            key=lambda item: (item.get("page_no", 10**9), item.get("line_no", 10**9)),
        )
        header_lines = [line for line in lines if (line.get("page_no") or 0) <= 3][:260]

        cover_patterns: list[tuple[re.Pattern[str], callable]] = [
            (
                re.compile(
                    rf"\bDate of Report\s*\(Date of earliest event reported\)\s*:?\s*"
                    rf"({date_re})(?:\s*\(\s*({date_re})\s*\))?",
                    re.IGNORECASE,
                ),
                lambda m: f"Reporting period: Date of earliest event reported: {m.group(2) or m.group(1)}",
            ),
            (
                re.compile(
                    rf"\bDate of earliest event reported\s*:?\s*({date_re})\b",
                    re.IGNORECASE,
                ),
                lambda m: f"Reporting period: Date of earliest event reported: {m.group(1)}",
            ),
            (
                re.compile(
                    rf"\bEvent date\s*:?\s*({date_re})\b",
                    re.IGNORECASE,
                ),
                lambda m: f"Reporting period: Event date: {m.group(1)}",
            ),
            (
                re.compile(
                    rf"\bFor the quarterly period ended\s+({date_re})\b",
                    re.IGNORECASE,
                ),
                lambda m: f"Reporting period: Quarterly period ended {m.group(1)}",
            ),
            (
                re.compile(
                    rf"\bFor the quarter ended\s+({date_re})\b",
                    re.IGNORECASE,
                ),
                lambda m: f"Reporting period: Quarter ended {m.group(1)}",
            ),
            (
                re.compile(
                    rf"\bFor the fiscal year ended\s+({date_re})\b",
                    re.IGNORECASE,
                ),
                lambda m: f"Reporting period: Fiscal year ended {m.group(1)}",
            ),
            (
                re.compile(
                    rf"\bFor the year ended\s+({date_re})\b",
                    re.IGNORECASE,
                ),
                lambda m: f"Reporting period: Year ended {m.group(1)}",
            ),
            (
                re.compile(
                    rf"\bFor the six months ended\s+({date_re})\b",
                    re.IGNORECASE,
                ),
                lambda m: f"Reporting period: Six months ended {m.group(1)}",
            ),
            (
                re.compile(
                    rf"\bFor the nine months ended\s+({date_re})\b",
                    re.IGNORECASE,
                ),
                lambda m: f"Reporting period: Nine months ended {m.group(1)}",
            ),
            (
                re.compile(
                    r"\bFor the transition period from\s+(.+?)\s+to\s+(.+?)(?:[.;]|$)",
                    re.IGNORECASE,
                ),
                lambda m: f"Reporting period: For the transition period from {norm(m.group(1))} to {norm(m.group(2))}",
            ),
        ]

        table_patterns: list[tuple[re.Pattern[str], callable]] = [
            (
                re.compile(
                    rf"\bQuarters?\s+Ended\s+({date_re})(?:\s+({date_re}))?\b",
                    re.IGNORECASE,
                ),
                lambda m: f"Reporting period: Current quarter ended {m.group(1)}",
            ),
            (
                re.compile(
                    rf"\bThree Months Ended\s+({date_re})(?:\s+(?:and\s+)?({date_re}))?\b",
                    re.IGNORECASE,
                ),
                lambda m: f"Reporting period: Current quarter ended {m.group(1)}",
            ),
        ]

        cover_lines = [line for line in header_lines if (line.get("page_no") or 0) <= 1][:120]
        cover_text = norm("\n".join(line.get("text", "") for line in cover_lines))
        header_text = norm("\n".join(line.get("text", "") for line in header_lines))
        if not header_text:
            header_text = norm((doc.get("text") or "")[:30000])

        for pattern, builder in cover_patterns:
            match = pattern.search(cover_text or header_text)
            if match:
                return [make_span(builder(match))]

        scan_lines = cover_lines or header_lines

        for idx, line in enumerate(scan_lines):
            for width in range(3, 0, -1):
                if idx + width > len(scan_lines):
                    break
                pieces = [
                    norm(scan_lines[pos].get("text", ""))
                    for pos in range(idx, idx + width)
                    if norm(scan_lines[pos].get("text", ""))
                ]
                joined = norm(" ".join(pieces))
                if not joined:
                    continue
                for pattern, builder in cover_patterns:
                    match = pattern.search(joined)
                    if match:
                        return [make_span(builder(match), line)]

        for pattern, builder in table_patterns:
            match = pattern.search(header_text)
            if match:
                return [make_span(builder(match))]

        for idx, line in enumerate(header_lines):
            for width in range(3, 0, -1):
                if idx + width > len(header_lines):
                    break
                pieces = [
                    norm(header_lines[pos].get("text", ""))
                    for pos in range(idx, idx + width)
                    if norm(header_lines[pos].get("text", ""))
                ]
                joined = norm(" ".join(pieces))
                if not joined:
                    continue
                for pattern, builder in table_patterns:
                    match = pattern.search(joined)
                    if match:
                        return [make_span(builder(match), line)]

        doc_name = str(doc.get("doc_name") or "")
        date_match = re.search(r"dated[-_]?(\d{4}-\d{2}-\d{2})", doc_name, re.IGNORECASE)
        if date_match and "8K" in doc_name.upper():
            return [make_span(format_doc_date(date_match.group(1)))]

        return []
    except Exception:
        return []
