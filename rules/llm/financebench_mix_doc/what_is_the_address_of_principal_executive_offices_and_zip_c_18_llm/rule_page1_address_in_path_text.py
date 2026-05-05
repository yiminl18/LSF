def rule_page1_address_in_path_text(doc: dict) -> list[dict]:
    """Match spans whose path_text itself contains a street-number address."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = (((span.get("structure") or {}).get("path_text")) or "")
            if span.get("page_no") == 1 and re.search(r'\|\s*\d{1,5}\s', path):
                out.append(span)
        return out
    except Exception:
        return []
