def rule_item1_business_telephone_sentence(doc: dict) -> list[dict]:
    """Match Item 1/Business narrative spans containing a sentence with the company's telephone number."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = span.get("structure", {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if re.search(r"item\s*1|business", path, re.I) and re.search(r"(our|the company[’'`s]?|adobe|amazon)[^\.]{0,80}telephone number", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
