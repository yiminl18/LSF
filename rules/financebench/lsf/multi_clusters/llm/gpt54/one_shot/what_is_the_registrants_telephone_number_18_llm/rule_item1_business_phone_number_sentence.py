def rule_item1_business_phone_number_sentence(doc: dict) -> list[dict]:
    """Match Item 1/Business narrative spans containing a phone number near 'telephone number'."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = span.get("structure", {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if re.search(r"item\s*1|business|overview", path, re.I) and re.search(r"telephone number.{0,40}(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s\-]?)\d{3,4}[\s\-]?\d{4,}", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
