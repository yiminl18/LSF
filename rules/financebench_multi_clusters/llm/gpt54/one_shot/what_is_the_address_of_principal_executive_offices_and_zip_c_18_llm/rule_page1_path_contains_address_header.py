def rule_page1_path_contains_address_header(doc: dict) -> list[dict]:
    """Match spans whose structural path already contains the address text as a header."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = (((span.get("structure") or {}).get("path_text")) or "")
            if span.get("page_no") == 1 and re.search(r'^\D*.*\|\s*\d{1,5}\s', path):
                out.append(span)
            elif span.get("page_no") == 1 and re.search(r'\|\s*[^|]*\b\d{5}(?:-\d{4})?\b', path):
                out.append(span)
        return out
    except Exception:
        return []
