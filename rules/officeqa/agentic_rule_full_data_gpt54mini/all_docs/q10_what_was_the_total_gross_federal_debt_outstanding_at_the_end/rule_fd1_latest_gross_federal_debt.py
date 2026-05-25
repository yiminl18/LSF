import re


_FD1_HEADING_RE = re.compile(r"\bTABLE\s+FD-1\b.*\bSummary of Federal Debt\b", re.IGNORECASE)
_YEAR_ROW_RE = re.compile(r"^\s*(?:19|20)\d{2}\s*(?:\.{2,}|\s+)?$", re.IGNORECASE)
_YEAR_ROW_WITH_SUFFIX_RE = re.compile(
    r"^\s*(?:19|20)\d{2}\s*(?:[-–]\s*[A-Za-z]{3,9}|\.{2,}.*)?$",
    re.IGNORECASE,
)
_MONTH_TOKEN_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b",
    re.IGNORECASE,
)
_NUMERIC_LINE_RE = re.compile(r"^\s*\$?(?:\d{4,}|\d{1,3}(?:,\d{3})+)(?:\.\d+)?\s*$")
_FISCAL_SENTENCE_RE = re.compile(
    r"At the end of (?:FY|fiscal year)\s+\d{4}.*?gross federal debt.*?(?:\$[\d,]+(?:\.\d+)?\s*(?:billion|trillion))?",
    re.IGNORECASE,
)


def rule_fd1_latest_gross_federal_debt(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def clean_number(text: str) -> str:
            cleaned = norm(text)
            cleaned = cleaned.replace("$", "")
            cleaned = cleaned.rstrip(".")
            return cleaned

        def add_span(spans: list[dict], seen: set[str], text: str, source: dict | None = None) -> None:
            cleaned = norm(text)
            if not cleaned:
                return
            key = cleaned.lower()
            if key in seen:
                return
            seen.add(key)
            span = {"text": cleaned}
            if source is not None:
                if source.get("page_no") is not None:
                    span["page_no"] = source.get("page_no")
                if source.get("line_no") is not None:
                    span["line_no"] = source.get("line_no")
                if source.get("paragraph_no") is not None:
                    span["paragraph_no"] = source.get("paragraph_no")
            spans.append(span)

        spans: list[dict] = []
        seen: set[str] = set()
        doc_name = (doc.get("doc_name") or "").lower()

        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]

        # Prefer the FD-1 table because it carries the exact millions value.
        table_start = None
        for idx, item in enumerate(lines):
            text = norm(item.get("text") or "")
            if _FD1_HEADING_RE.search(text):
                table_start = idx
                break

        if table_start is not None:
            latest_value = None
            latest_source = None
            current_label = False
            captured_value = False
            # Scan FD-1 in order and keep the latest fiscal-year row from the first block.
            for idx in range(table_start, min(table_start + 1200, len(lines))):
                text = norm(lines[idx].get("text") or "")
                if not text:
                    continue

                if (
                    re.search(r"\bFD-2\b", text, re.IGNORECASE)
                    or (
                        re.search(r"\bcontinued\b", text, re.IGNORECASE)
                        and re.search(
                            r"\b(?:nonmarketable|marketable|public debt securities|government account series|corporations and other agencies)\b",
                            text,
                            re.IGNORECASE,
                        )
                    )
                ):
                    break

                if _YEAR_ROW_RE.match(text):
                    current_label = True
                    captured_value = False
                    continue

                if not current_label or captured_value:
                    continue

                if _NUMERIC_LINE_RE.match(text):
                    latest_value = clean_number(text)
                    latest_source = lines[idx]
                    captured_value = True

            if latest_value is not None:
                add_span(spans, seen, latest_value, latest_source)
        
        # Supplemental narrative sentence from the federal debt discussion.
        narrative_span = None
        for item in lines:
            text = norm(item.get("text") or "")
            if not text:
                continue
            if "gross federal debt" in text.lower() and (
                re.search(r"\bAt the end of (?:FY|fiscal year)\s+\d{4}\b", text, re.IGNORECASE)
                or re.search(r"\bAs of\s+\w+\s+\d{4}\b", text, re.IGNORECASE)
            ):
                narrative_span = {"text": text}
                if item.get("page_no") is not None:
                    narrative_span["page_no"] = item.get("page_no")
                if item.get("line_no") is not None:
                    narrative_span["line_no"] = item.get("line_no")
                break

        if narrative_span is not None:
            add_span(spans, seen, narrative_span["text"], narrative_span)

        if spans:
            return spans

        return []
    except Exception:
        return []
