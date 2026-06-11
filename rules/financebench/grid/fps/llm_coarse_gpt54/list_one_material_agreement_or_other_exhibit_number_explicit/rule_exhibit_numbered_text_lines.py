def rule_exhibit_numbered_text_lines(doc: dict) -> list[dict]:
    """Match text/list/header spans that begin with an exhibit number and description."""
    import re
    out = []
    pat = re.compile(r'^\s*Exhibit\s+\d+(\.\d+)?[A-Za-z]?\s*[:\-]', re.I)
    try:
        for span in doc.get("texts", []):
            if span.get("label") in {"text", "list_item", "section_header"}:
                text = span.get("text", "") or ""
                if pat.search(text):
                    out.append(span)
    except Exception:
        return []
    return out
