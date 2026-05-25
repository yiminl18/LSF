import re


_HEADING_RE = re.compile(
    r"\bTABLE\s+USCC-2\b.*\bAmounts Outstanding and in Circulation\b",
    re.IGNORECASE,
)
_COMPARATIVE_RE = re.compile(
    r"\bCOMPARATIVE TOTALS OF CURRENCY AND COIN IN CIRCULATION\b",
    re.IGNORECASE,
)
_NUMERIC_RE = re.compile(r"^\s*\$?\d[\d,]*(?:\.\d+)?\s*$")
_DECIMAL_RE = re.compile(r"^\s*\$?\d[\d,]*\.\d+\s*$")


def rule_uscc2_latest_per_capita_currency_and_coin(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def normalize_amount(text: str) -> str:
            cleaned = norm(text).replace("$", "")
            digits = re.sub(r"\D", "", cleaned)
            if len(digits) < 3:
                return cleaned
            whole = digits[:-2]
            frac = digits[-2:]
            try:
                whole_text = f"{int(whole):,}"
            except Exception:
                whole_text = whole.lstrip("0") or "0"
            return f"{whole_text}.{frac}"

        def add_span(spans: list[dict], seen: set[str], text: str, source: dict | None = None) -> None:
            cleaned = normalize_amount(text)
            if not cleaned:
                return
            key = cleaned.lower()
            if key in seen:
                return
            seen.add(key)
            span = {"text": cleaned}
            if source:
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
            if _HEADING_RE.search(text):
                heading_idx = idx
                break
        if heading_idx is None:
            return []

        comparative_idx = None
        scan_limit = min(len(lines), heading_idx + 350)
        for idx in range(heading_idx, scan_limit):
            text = norm(lines[idx].get("text") or "")
            if _COMPARATIVE_RE.search(text):
                comparative_idx = idx
                break
        if comparative_idx is None:
            return []

        # The per-capita figures are listed after the comparative totals table
        # and the accompanying footnotes. The first standalone decimal line
        # after that note block is the most recent reporting date.
        note_end_idx = None
        for idx in range(comparative_idx, min(len(lines), comparative_idx + 250)):
            text = norm(lines[idx].get("text") or "")
            lowered = text.lower()
            if "excludes coin sold to collectors at premium prices" in lowered:
                note_end_idx = idx
                break
            if "based on the bureau of the census estimates of population" in lowered:
                note_end_idx = idx
                break
            if lowered.startswith("source:"):
                note_end_idx = idx
                break
        if note_end_idx is None:
            note_end_idx = comparative_idx

        candidate_idx = None
        for idx in range(note_end_idx + 1, min(len(lines), note_end_idx + 80)):
            text = norm(lines[idx].get("text") or "").replace("$", "")
            if _DECIMAL_RE.match(text):
                candidate_idx = idx
                break

        if candidate_idx is None:
            # Fallback: look for the first decimal-like line after the heading.
            for idx in range(comparative_idx, min(len(lines), comparative_idx + 500)):
                text = norm(lines[idx].get("text") or "").replace("$", "")
                if _DECIMAL_RE.match(text):
                    candidate_idx = idx
                    break

        if candidate_idx is None:
            return []

        spans: list[dict] = []
        seen: set[str] = set()
        add_span(spans, seen, lines[candidate_idx].get("text") or "", lines[candidate_idx])
        return spans
    except Exception:
        return []
