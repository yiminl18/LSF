def rule_item1_overview_office_and_phone(doc: dict) -> list[dict]:
    """Match Item 1 overview spans mentioning executive offices and a phone number together."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = span.get("structure", {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if re.search(r"item\s*1|business|overview", path, re.I) and re.search(r"executive offices.*(?:telephone number|website)", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
