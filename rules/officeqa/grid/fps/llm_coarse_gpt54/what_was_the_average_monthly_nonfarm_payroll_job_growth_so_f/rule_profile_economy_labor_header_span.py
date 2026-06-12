def rule_profile_economy_labor_header_span(doc: dict) -> list[dict]:
    """Return labor-related section headers themselves, useful as anchors for downstream extraction."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "section_header"
            and re.search(r'(Employment and unemployment|Labor Markets and Wages|Labor Markets)', span.get("text") or "", re.I)
        ]
    except Exception:
        return []
