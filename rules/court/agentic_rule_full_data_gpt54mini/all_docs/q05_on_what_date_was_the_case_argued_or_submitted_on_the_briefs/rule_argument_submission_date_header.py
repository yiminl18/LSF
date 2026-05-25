from __future__ import annotations
import re


_ARGUED_AND_SUBMITTED_RE = re.compile(
    r"(?i)^\s*Argued and Submitted(?:\s+En Banc)?\s+"
    r"(?P<date>[A-Z][a-z]+ \d{1,2}, \d{4})(?:\s*\*+\s*)?\s*$"
)
_SUBMITTED_RE = re.compile(
    r"(?i)^\s*Submitted(?:\s+on the briefs)?(?:\s+En Banc)?\s+"
    r"(?P<date>[A-Z][a-z]+ \d{1,2}, \d{4})(?:\s*\*+\s*)?\s*$"
)


def rule_argument_submission_date_header(doc: dict) -> list[dict]:
    try:
        spans = []
        seen = set()

        def add_span(span_text: str, source_item: dict) -> None:
            cleaned = re.sub(r"\s+", " ", (span_text or "")).strip().rstrip(".,;:")
            if not cleaned:
                return
            key = cleaned.lower()
            if key in seen:
                return
            seen.add(key)
            span = {"text": cleaned}
            for field in ("page_no", "line_no", "paragraph_no"):
                if field in source_item:
                    span[field] = source_item[field]
            spans.append(span)

        def extract_date(text: str) -> str | None:
            normalized = re.sub(r"\s+", " ", (text or "")).strip()
            if not normalized:
                return None
            for pattern in (_ARGUED_AND_SUBMITTED_RE, _SUBMITTED_RE):
                match = pattern.match(normalized)
                if match:
                    return match.group("date")
            return None

        def scan_items(items: list[dict]) -> None:
            limit = min(len(items or []), 80)
            for i in range(limit):
                item = items[i] or {}
                text = (item.get("text") or "").strip()
                if not text:
                    continue

                if extract_date(text):
                    add_span(text, item)
                    return

                # Some OCR variants split the header across adjacent rows.
                if i + 1 < limit:
                    next_item = items[i + 1] or {}
                    next_text = (next_item.get("text") or "").strip()
                    if next_text:
                        combined = f"{text} {next_text}"
                        if extract_date(combined):
                            add_span(combined, item)
                            return

        scan_items(doc.get("lines") or [])
        if not spans:
            scan_items(doc.get("paragraphs") or [])

        if not spans:
            full_text = doc.get("text") or ""
            for pattern in (_ARGUED_AND_SUBMITTED_RE, _SUBMITTED_RE):
                match = pattern.search(full_text)
                if match:
                    add_span(match.group(0), {})
                    break

        return spans
    except Exception:
        return []
