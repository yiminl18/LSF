def rule_page1_area_code_keyword(doc: dict) -> list[dict]:
    """Match page-1 spans mentioning area code near registrant telephone text."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r"(telephone\s+number.*area\s+code|area\s+code.*telephone\s+number)", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
