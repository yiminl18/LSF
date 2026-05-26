import re


_HEADING_RE = re.compile(
    r"\bMS-1\b",
    re.IGNORECASE,
)
_DECIMAL_RE = re.compile(r"^\s*\$?\d[\d,.,\s]*\s*$")


def rule_ms1_latest_per_capita_currency_and_coin(doc: dict) -> list[dict]:
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
            if not _HEADING_RE.search(text):
                continue
            window_end = min(len(lines), idx + 220)
            if any("per capita" in norm(lines[j].get("text") or "").lower() for j in range(idx, window_end)):
                heading_idx = idx
        if heading_idx is None:
            return []

        source_idx = None
        for idx in range(heading_idx, min(len(lines), heading_idx + 700)):
            text = norm(lines[idx].get("text") or "")
            if text.lower().startswith("source:"):
                source_idx = idx
                break
        if source_idx is None:
            source_idx = min(len(lines), heading_idx + 700)

        candidate_idx = None
        for idx in range(heading_idx, source_idx):
            text = norm(lines[idx].get("text") or "").replace("$", "")
            if _DECIMAL_RE.match(text) and "." in text:
                candidate_idx = idx
        if candidate_idx is None:
            return []

        spans: list[dict] = []
        seen: set[str] = set()
        add_span(spans, seen, lines[candidate_idx].get("text") or "", lines[candidate_idx])
        return spans
    except Exception:
        return []
