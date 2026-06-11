def rule_our_telephone_number_in_item1_business(doc: dict) -> list[dict]:
    """Match Item 1/Business narrative spans stating 'Our telephone number is ...'."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if re.search(r"Our telephone number is", text, re.I) and re.search(r"Item 1|Business", path, re.I):
                out.append(span)
        return out
    except Exception:
        return []
