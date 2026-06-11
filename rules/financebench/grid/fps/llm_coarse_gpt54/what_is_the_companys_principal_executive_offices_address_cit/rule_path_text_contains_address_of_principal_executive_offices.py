def rule_path_text_contains_address_of_principal_executive_offices(doc: dict) -> list[dict]:
    """Match spans whose structural path_text contains an address span tied to principal executive offices."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = (((span.get("structure") or {}).get("path_text")) or "")
            if re.search(r'address of principal executive offices', path, re.I):
                out.append(span)
        return out
    except Exception:
        return []
