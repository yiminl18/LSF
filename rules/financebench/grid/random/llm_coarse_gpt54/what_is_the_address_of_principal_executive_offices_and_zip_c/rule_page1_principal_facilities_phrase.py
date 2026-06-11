def rule_page1_principal_facilities_phrase(doc: dict) -> list[dict]:
    """Match spans mentioning executive offices and principal facilities, often in Item 1."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if re.search(r'executive offices.*principal facilities.*located at', (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
