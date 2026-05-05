def rule_item1_business_path_telephone_keyword(doc: dict) -> list[dict]:
    """Match any span under Item 1/Business path containing a telephone keyword."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = span.get("structure", {}).get("path_text", "") or ""
            blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
            if re.search(r"item\s*1|business", path, re.I) and re.search(r"telephone number", blob, re.I):
                out.append(span)
        return out
    except Exception:
        return []
