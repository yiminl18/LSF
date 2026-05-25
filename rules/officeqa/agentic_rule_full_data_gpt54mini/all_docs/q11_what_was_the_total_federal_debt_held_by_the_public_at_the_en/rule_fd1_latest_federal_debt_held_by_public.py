import re


_FD1_HEADING_RE = re.compile(r"\bTABLE\s+FD-1\b.*\bSummary of Federal Debt\b", re.IGNORECASE)
_ANNUAL_ROW_RE = re.compile(r"^\s*((?:19|20)\d{2})\s*\.{2,}\s*$")
_MONTH_ROW_RE = re.compile(
    r"^\s*(?:"
    r"(?:19|20)\d{2}\s*-\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[A-Za-z]*"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[A-Za-z]*"
    r")\b",
    re.IGNORECASE,
)
_NUMERIC_RE = re.compile(r"^\s*\$?(?:\d{1,3}(?:,\d{3})+|\d{4,})(?:\.\d+)?\s*$")


def rule_fd1_latest_federal_debt_held_by_public(doc: dict) -> list[dict]:
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

        heading_idx = None
        for idx, item in enumerate(lines):
            text = norm(item.get("text") or "")
            if _FD1_HEADING_RE.search(text):
                heading_idx = idx
                break
        if heading_idx is None:
            return []

        annual_rows: list[tuple[int, int]] = []
        i = heading_idx + 1
        scan_limit = min(len(lines), heading_idx + 2500)
        while i < scan_limit:
            text = norm(lines[i].get("text") or "")
            if not text:
                i += 1
                continue

            if _MONTH_ROW_RE.match(text):
                if annual_rows:
                    break
                i += 1
                continue

            if _ANNUAL_ROW_RE.match(text):
                numeric_cells: list[tuple[int, str]] = []
                for j in range(i + 1, min(scan_limit, i + 20)):
                    candidate = norm(lines[j].get("text") or "")
                    if _NUMERIC_RE.match(candidate):
                        numeric_cells.append((j, candidate.replace("$", "")))
                        if len(numeric_cells) >= 9:
                            break

                if len(numeric_cells) >= 6:
                    target_idx, _ = numeric_cells[5]
                    annual_rows.append((i, target_idx))
                    i = target_idx + 1
                    continue

            i += 1

        if not annual_rows:
            return []

        label_idx, value_idx = annual_rows[-1]
        label_text = norm(lines[label_idx].get("text") or "")
        value_text = norm(lines[value_idx].get("text") or "").replace("$", "")

        spans: list[dict] = []
        seen: set[str] = set()
        add_span(spans, seen, value_text, lines[value_idx])
        add_span(spans, seen, f"{label_text} {value_text}", lines[label_idx])
        return spans
    except Exception:
        return []
