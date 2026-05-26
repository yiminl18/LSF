import re
from typing import List, Dict, Any


def rule_reporting_period(doc: dict) -> list[dict]:
    try:
        text = doc.get("text") or ""
        lines = doc.get("lines") or []

        def norm_ws(value: str) -> str:
            return re.sub(r"\s+", " ", value or "").strip()

        def make_span(raw_text: str, source: dict | None = None) -> dict:
            span = {"text": norm_ws(raw_text)}
            if source:
                if "page_no" in source and source["page_no"] is not None:
                    span["page_no"] = source["page_no"]
                if "line_no" in source and source["line_no"] is not None:
                    span["line_no"] = source["line_no"]
            return span

        def add_span(spans: list[dict], seen: set[str], raw_text: str, source: dict | None = None) -> None:
            cleaned = norm_ws(raw_text)
            if not cleaned or cleaned in seen:
                return
            seen.add(cleaned)
            spans.append(make_span(cleaned, source))

        spans: list[dict] = []
        seen: set[str] = set()

        # Work from the beginning of the filing; the cover page usually contains the answer.
        scan_lines = lines[:220] if lines else []
        scan_text = norm_ws("\n".join(str(line.get("text", "")) for line in scan_lines))
        if not scan_text:
            scan_text = norm_ws(text[:30000])

        # 8-K / current report pattern.
        date_re = r"([A-Za-z.]+\s+\d{1,2},\s+\d{4})"

        event_patterns = [
            rf"Date of Report \(Date of earliest event reported\)\s*:?\s*{date_re}(?:\s*\(\s*{date_re}\s*\))?",
            rf"Date of earliest event reported\s*:?\s*{date_re}(?:\s*\(\s*{date_re}\s*\))?",
            rf"{date_re}\s+Date of Report \(Date of earliest event reported\)",
            rf"{date_re}\s+Date of earliest event reported",
        ]
        for pat in event_patterns:
            m = re.search(pat, scan_text, flags=re.IGNORECASE)
            if m:
                # Prefer the event date if it is parenthesized after the report date.
                event_date = None
                for idx in range(m.lastindex or 0, 0, -1):
                    candidate = m.group(idx)
                    if candidate and re.fullmatch(date_re, candidate, flags=re.IGNORECASE):
                        event_date = candidate
                        break
                add_span(spans, seen, event_date or m.group(0))
                break

        # Cover-page reporting period patterns for 10-Q / 10-K filings.
        period_patterns = [
            rf"For the quarterly period ended\s+{date_re}",
            rf"For the quarter ended\s+{date_re}",
            rf"For the fiscal year ended\s+{date_re}",
            rf"For the year ended\s+{date_re}",
            rf"For the six months ended\s+{date_re}",
            rf"For the nine months ended\s+{date_re}",
            rf"(?:Quarters?|Three Months|Three-Months)\s+Ended\s+{date_re}",
            rf"(?:Quarter|Three Months|Three-Months)\s+Ended\s+{date_re}",
            r"For the transition period from\s+([A-Z0-9, .\-/]+?)\s+to\s+([A-Z0-9, .\-/]+?)\.",
            r"For the transition period from\s+([A-Z0-9, .\-/]+?)\s+to\s+([A-Z0-9, .\-/]+?)",
        ]
        for pat in period_patterns:
            m = re.search(pat, scan_text, flags=re.IGNORECASE)
            if not m:
                continue
            raw = m.group(0)
            lower = raw.lower()
            if "transition period" in lower:
                add_span(spans, seen, raw)
            else:
                period_date = None
                for idx in range(1, (m.lastindex or 0) + 1):
                    candidate = m.group(idx)
                    if candidate and re.fullmatch(date_re, candidate, flags=re.IGNORECASE):
                        period_date = candidate
                        break
                if period_date:
                    if "fiscal year" in lower or re.search(r"\byear ended\b", lower):
                        add_span(spans, seen, f"fiscal year ended {period_date}")
                    elif "quarter" in lower or "three months" in lower or "quarters" in lower:
                        add_span(spans, seen, f"quarter ended {period_date}")
                    elif "six months" in lower:
                        add_span(spans, seen, f"six months ended {period_date}")
                    elif "nine months" in lower:
                        add_span(spans, seen, f"nine months ended {period_date}")
                    else:
                        add_span(spans, seen, raw)
                else:
                    add_span(spans, seen, raw)
            break

        # Some exhibit-style filings only expose the period in a financial table header.
        if not spans:
            table_patterns = [
                rf"Quarters?\s+Ended\s+{date_re}",
                rf"Three Months Ended\s+{date_re}",
                rf"Months Ended\s+{date_re}",
            ]
            for pat in table_patterns:
                m = re.search(pat, scan_text, flags=re.IGNORECASE)
                if not m:
                    continue
                period_date = None
                for idx in range(1, (m.lastindex or 0) + 1):
                    candidate = m.group(idx)
                    if candidate and re.fullmatch(date_re, candidate, flags=re.IGNORECASE):
                        period_date = candidate
                        break
                if period_date:
                    add_span(spans, seen, f"quarter ended {period_date}")
                else:
                    add_span(spans, seen, m.group(0))
                break

        # Fallback: search line-by-line for a likely reporting-period header.
        if not spans:
            fallback_patterns = [
                r"\b(for the (?:quarterly period|quarter|fiscal year|year|six months|nine months) ended\b.*)",
                r"\b(date of report \(date of earliest event reported\):\b.*)",
                r"\b(date of earliest event reported:\b.*)",
            ]
            for line in scan_lines:
                line_text = norm_ws(line.get("text", ""))
                if not line_text:
                    continue
                for pat in fallback_patterns:
                    if re.search(pat, line_text, flags=re.IGNORECASE):
                        add_span(spans, seen, line_text, line)
                        break
                if spans:
                    break

        # If the filing is an 8-K and the text is too messy, fall back to the filing date in the doc name.
        if not spans:
            doc_name = str(doc.get("doc_name") or "")
            m = re.search(r"(?:dated[-_]?)(\d{4}-\d{2}-\d{2})", doc_name, flags=re.IGNORECASE)
            if m:
                add_span(spans, seen, m.group(1))

        # Final fallback: if nothing matched, return an empty list.
        return spans
    except Exception:
        return []
