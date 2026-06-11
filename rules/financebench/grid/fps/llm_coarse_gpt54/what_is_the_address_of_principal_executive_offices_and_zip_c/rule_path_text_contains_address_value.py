def rule_path_text_contains_address_value(doc: dict) -> list[dict]:
    """Match spans whose structure.path_text contains an address-like heading value near the registrant block."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r'\b(address of principal executive offices)\b', path, re.I):
                out.append(span)
                continue
            if re.search(r'\b\d{1,6}\s+\S+', path) and re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', path, re.I):
                out.append(span)
        return out
    except Exception:
        return []
