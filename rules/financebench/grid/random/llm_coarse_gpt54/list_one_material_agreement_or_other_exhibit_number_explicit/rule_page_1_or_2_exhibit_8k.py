def rule_page_1_or_2_exhibit_8k(doc: dict) -> list[dict]:
    """Match early-page 8-K exhibit references, common in short current reports."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            page = span.get("page_no") or 0
            txt = span.get("text") or ""
            if page in (1, 2, 3) and re.search(r"\b(exhibit|item\s*9\.?01)\b", txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
