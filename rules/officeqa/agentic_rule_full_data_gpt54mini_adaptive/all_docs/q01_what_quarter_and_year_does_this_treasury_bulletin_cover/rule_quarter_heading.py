import re


QUARTER_RE = re.compile(
    r"\b(?:first|second|third|fourth)[- ]quarter\b",
    re.IGNORECASE,
)


def rule_quarter_heading(doc: dict) -> list[dict]:
    try:
        spans = []
        seen = set()

        for line in doc.get("lines") or []:
            text = (line.get("text") or "").strip()
            if not text or not QUARTER_RE.search(text):
                continue
            lowered = text.lower()
            if "treasury bulletin" not in lowered:
                continue
            key = (line.get("page_no"), line.get("line_no"), text)
            if key in seen:
                continue
            seen.add(key)
            span = {"text": text}
            if line.get("page_no") is not None:
                span["page_no"] = line["page_no"]
            if line.get("line_no") is not None:
                span["line_no"] = line["line_no"]
            spans.append(span)

        return spans
    except Exception:
        return []
