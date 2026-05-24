def rule_item1_business_path_phone_number(doc: dict) -> list[dict]:
    """Match any span under Item 1/Business path containing a phone number."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = span.get("structure", {}).get("path_text", "") or ""
            blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
            if re.search(r"item\s*1|business", path, re.I) and re.search(r"(?:\+?\d{1,3}\s*)?(?:\(?\d{3}\)?[\s\-]?)\d{3,4}[\s\-]?\d{4,}", blob):
                out.append(span)
        return out
    except Exception:
        return []
