def rule_item1_overview_phone_number_near_website(doc: dict) -> list[dict]:
    """Match Item 1 overview spans where the phone number appears near the company website."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = span.get("structure", {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if re.search(r"item\s*1|business|overview", path, re.I) and re.search(r"telephone number.{0,60}website|website.{0,60}telephone number", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
