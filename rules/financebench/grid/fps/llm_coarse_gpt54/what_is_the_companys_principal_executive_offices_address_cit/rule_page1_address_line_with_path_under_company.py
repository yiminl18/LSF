def rule_page1_address_line_with_path_under_company(doc: dict) -> list[dict]:
    """Match page-1 address-like spans under the top company block on the cover page."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            path = (((span.get("structure") or {}).get("path_text")) or "")
            text = (span.get("text") or "").strip()
            if path and not re.search(r'item\s+\d', path, re.I):
                if re.search(r'\d{1,6}\s+\S+', text) and (
                    re.search(r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b', text) or
                    re.search(r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+', text) or
                    re.search(r'\bUnited Kingdom\b', text)
                ):
                    out.append(span)
        return out
    except Exception:
        return []
