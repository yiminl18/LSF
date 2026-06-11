def rule_item1_overview_telephone_and_website_after_address(doc: dict) -> list[dict]:
    """Match Item 1 overview text where address is followed by telephone number and website."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if re.search(r'located at .*?telephone number is .*?website is', (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
