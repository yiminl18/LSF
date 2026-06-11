def rule_page1_address_from_path_text_value(doc: dict) -> list[dict]:
    """Match spans whose path_text itself is the address line under the company block."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = (((span.get("structure") or {}).get("path_text")) or "").strip()
            if span.get("page_no") == 1 and re.search(r'\d{1,6}\s+\S+', path):
                if re.search(r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b', path) or re.search(r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+', path) or re.search(r'\bUnited Kingdom\b', path):
                    out.append(span)
        return out
    except Exception:
        return []
