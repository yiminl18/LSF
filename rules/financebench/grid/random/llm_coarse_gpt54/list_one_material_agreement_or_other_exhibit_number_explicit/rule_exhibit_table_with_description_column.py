def rule_exhibit_table_with_description_column(doc: dict) -> list[dict]:
    """Match tables that look like exhibit lists with 'Exhibit No.' and 'Description' columns."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = span.get("text") or ""
            if re.search(r"\bexhibit no\.?\b", txt, re.I) and re.search(r"\bdescription\b", txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
