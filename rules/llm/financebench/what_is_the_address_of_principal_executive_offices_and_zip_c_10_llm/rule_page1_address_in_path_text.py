def rule_page1_address_in_path_text(doc: dict) -> list[dict]:
    """Match spans whose structure.path_text itself contains an address-like string."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for span in spans:
            path = ((span.get("structure") or {}).get("path_text") or "")
            low = path.lower()
            if re.search(r"\b\d{1,6}\b", path) and re.search(r"\b(avenue|drive|road|plaza|street|way)\b", low):
                out.append(span)
            elif re.search(r"\b\d{5}(?:-\d{4})?\b", path):
                out.append(span)
        return out
    except Exception:
        return []
