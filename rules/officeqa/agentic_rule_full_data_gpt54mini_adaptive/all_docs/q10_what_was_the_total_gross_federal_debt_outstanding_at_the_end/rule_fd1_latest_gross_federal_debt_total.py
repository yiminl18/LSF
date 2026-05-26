import re


_FD1_HEADING_RE = re.compile(r"\bTABLE\s+FD-1\b.*\bSummary of Federal Debt\b", re.IGNORECASE)
_YEAR_ROW_RE = re.compile(r"^\s*(?:19|20)\d{2}\s*(?:\.{2,}|\s+)?$", re.IGNORECASE)
_NUMERIC_LINE_RE = re.compile(r"^\s*\$?(?:\d{1,3}(?:,\d{3})+|\d{4,})(?:\.\d+)?\s*$")
_TARGET_DOC_RE = re.compile(
    r"\bgross federal debt\b.*\b(?:FY|fiscal year)\s+2024\b|\b(?:FY|fiscal year)\s+2024\b.*\bgross federal debt\b",
    re.IGNORECASE,
)


def rule_fd1_latest_gross_federal_debt_total(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

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

        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        if not lines:
            return []

        # Restrict to the recent bulletins that discuss FY 2024 gross federal debt.
        full_text = norm(doc.get("text") or " ".join(norm(item.get("text") or "") for item in lines))
        target_hit = bool(full_text and _TARGET_DOC_RE.search(full_text))
        if not target_hit:
            return []

        heading_idx = None
        for idx, item in enumerate(lines):
            text = norm(item.get("text") or "")
            if _FD1_HEADING_RE.search(text):
                heading_idx = idx
                break
        if heading_idx is None:
            return []

        latest_value = None
        latest_source = None
        in_annual_block = False
        captured_value_for_current_row = False

        scan_limit = min(len(lines), heading_idx + 1500)
        for idx in range(heading_idx + 1, scan_limit):
            text = norm(lines[idx].get("text") or "")
            if not text:
                continue

            if re.search(r"\bFD-2\b", text, re.IGNORECASE):
                break

            if _YEAR_ROW_RE.match(text):
                in_annual_block = True
                captured_value_for_current_row = False
                continue

            if not in_annual_block:
                continue

            if _NUMERIC_LINE_RE.match(text):
                if not captured_value_for_current_row:
                    latest_value = text.replace("$", "")
                    latest_source = lines[idx]
                    captured_value_for_current_row = True
                continue

            if captured_value_for_current_row and re.search(r"[A-Za-z]", text):
                break

        if latest_value is None:
            return []

        spans: list[dict] = []
        seen: set[str] = set()
        add_span(spans, seen, latest_value, latest_source)
        return spans
    except Exception:
        return []
