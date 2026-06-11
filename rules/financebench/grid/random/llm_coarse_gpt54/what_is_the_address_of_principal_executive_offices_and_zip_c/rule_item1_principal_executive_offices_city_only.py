def rule_item1_principal_executive_offices_city_only(doc: dict) -> list[dict]:
    """Match Item 1 text mentioning principal executive offices are located, even if only city/state is given."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if re.search(r'principal executive offices are located', (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
