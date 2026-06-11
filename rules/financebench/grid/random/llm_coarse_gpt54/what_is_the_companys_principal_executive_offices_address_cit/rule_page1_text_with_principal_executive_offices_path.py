def rule_page1_text_with_principal_executive_offices_path(doc: dict) -> list[dict]:
    """Match spans whose path_text itself contains an address-like header and principal-office context."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            if span.get("page_no") != 1:
                continue
            if re.search(r'principal executive offices', path, re.I) or (
                re.search(r'\b\d+\s+\S+', path) and re.search(r'\|', path)
            ):
                out.append(span)
        return out
    except Exception:
        return []
