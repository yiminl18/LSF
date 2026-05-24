def rule_item1_overview_phone_number_only_in_sentence(doc: dict) -> list[dict]:
    """Match overview/business spans with a phone number in a narrative sentence."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = span.get("structure", {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if re.search(r"overview|item 1|business", path, re.I) and re.search(r"(?:our|the)\s+telephone number\s+is\s+(?:\+?\d{1,3}[\s-]?)?(?:\d{3}|\(\d{3}\))[\s\-]?\d{3,4}[\s\-]?\d{4,}", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
