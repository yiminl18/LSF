import re


_FD6_HEADING_RE = re.compile(
    r"\bTABLE\s+FD-6\b.*\bDebt Subject to Statutory Limit\b",
    re.IGNORECASE,
)
_ROW_RE = re.compile(
    r"^\s*(?:"
    r"(?:\d{4}\s*[–-]\s*)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b"
    r"|(?:\d{4}\s*)?\.{2,}\s*$"
    r")",
    re.IGNORECASE,
)
_MAJOR_TABLE_RE = re.compile(r"^\s*TABLE\s+FD-\d+\b", re.IGNORECASE)


def rule_fd6_latest_statutory_debt_limit(doc: dict) -> list[dict]:
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
            if _FD6_HEADING_RE.search(norm(item.get("text") or "")):
                heading_idx = idx
                break
        if heading_idx is None:
            return []

        row_indices: list[int] = []
        limit = min(len(lines), heading_idx + 2500)
        for idx in range(heading_idx + 1, limit):
            text = norm(lines[idx].get("text") or "")
            if not text:
                continue
            if _MAJOR_TABLE_RE.match(text) and idx > heading_idx + 5:
                break
            if _ROW_RE.match(text):
                row_indices.append(idx)

        if not row_indices:
            return []

        row_idx = row_indices[-1]
        block_lines = [norm(lines[row_idx].get("text") or "")]
        for j in range(row_idx + 1, min(len(lines), row_idx + 8)):
            text = norm(lines[j].get("text") or "")
            if not text:
                continue
            if _ROW_RE.match(text) or _MAJOR_TABLE_RE.match(text):
                break
            block_lines.append(text)

        block_text = " ".join(block_lines)
        if not block_text:
            return []

        spans: list[dict] = []
        seen: set[str] = set()
        add_span(spans, seen, block_text, lines[row_idx])

        row_text = norm(lines[row_idx].get("text") or "")
        if row_text and row_text.lower() != block_text.lower():
            add_span(spans, seen, row_text, lines[row_idx])

        return spans
    except Exception:
        return []
