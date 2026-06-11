def rule_item1_overview_principal_facilities_address(doc: dict) -> list[dict]:
    """Match overview text where executive offices and principal facilities are located at a full address."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if re.search(r'executive offices and principal facilities are located at', (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
