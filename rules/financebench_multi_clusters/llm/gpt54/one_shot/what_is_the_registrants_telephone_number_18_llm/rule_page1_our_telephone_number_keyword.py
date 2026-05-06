def rule_page1_our_telephone_number_keyword(doc: dict) -> list[dict]:
    """Match Item 1 business text spans that say 'Our telephone number is'."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            path = span.get("structure", {}).get("path_text", "") or ""
            if re.search(r"\bour telephone number is\b", text, re.I) and re.search(r"item 1|business", path, re.I):
                out.append(span)
        return out
    except Exception:
        return []
