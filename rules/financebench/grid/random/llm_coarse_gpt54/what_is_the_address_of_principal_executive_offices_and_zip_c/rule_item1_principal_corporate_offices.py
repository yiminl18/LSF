def rule_item1_principal_corporate_offices(doc: dict) -> list[dict]:
    """Match Item 1 text saying principal corporate offices are located in a city/state."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if re.search(r'principal corporate offices are located', (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
