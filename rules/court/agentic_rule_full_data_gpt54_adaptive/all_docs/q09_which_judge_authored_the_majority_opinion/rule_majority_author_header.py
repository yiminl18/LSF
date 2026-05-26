import re


_HEADER_RE = re.compile(
    r"\b(?:(?:Order|Memorandum)\s*;\s*)?(?:Amended\s+)?Opinion by (?:Chief )?Judge\b[^\n]*",
    re.IGNORECASE,
)
_PER_CURIAM_RE = re.compile(r"\bPer Curiam\b", re.IGNORECASE)


def rule_majority_author_header(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def span_with_meta(item: dict, text: str | None = None) -> dict:
            span = {"text": text if text is not None else norm(item.get("text") or "")}
            for key in ("page_no", "line_no", "paragraph_no"):
                if key in item:
                    span[key] = item[key]
            return span

        for key, limit in (("lines", 260), ("paragraphs", 140), ("pages", 20)):
            items = [item for item in (doc.get(key) or []) if isinstance(item, dict)]
            for item in items[:limit]:
                text = norm(item.get("text") or "")
                if not text:
                    continue
                header_match = _HEADER_RE.search(text)
                if header_match:
                    return [span_with_meta(item, header_match.group(0).strip(" ;."))]
                if _PER_CURIAM_RE.search(text):
                    if text.upper().startswith("PER CURIAM"):
                        return [span_with_meta(item, "PER CURIAM authored the majority opinion.")]
                    if text.lower() in {"per curiam opinion", "per curiam"}:
                        return [span_with_meta(item, "PER CURIAM authored the majority opinion.")]

        text = doc.get("text") or ""
        header_match = _HEADER_RE.search(text[:12000])
        if header_match:
            return [{"text": header_match.group(0).strip(" ;.")}]

        if _PER_CURIAM_RE.search(text[:12000]):
            return [{"text": "PER CURIAM authored the majority opinion."}]

        return []
    except Exception:
        return []
