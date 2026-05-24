def rule_page1_address_header_path_and_child_label(doc: dict) -> list[dict]:
    """Match child spans under an address header path that contain address/zip labels."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            path = (((span.get("structure") or {}).get("path_text")) or "")
            text = (span.get("text") or "")
            if re.search(r'\|\s*\d{1,5}\s', path) and re.search(r'address|zip code|telephone number', text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
