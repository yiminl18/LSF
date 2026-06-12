def rule_esf_contents_section(doc: dict) -> list[dict]:
    """Match section headers named Exchange Stabilization Fund, often in contents pages."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "section_header"
            and re.search(r'EXCHANGE STABILIZATION FUND', span.get("text", ""), re.I)
        ]
    except Exception:
        return []
