def rule_item1_principal_corporate_offices(doc: dict) -> list[dict]:
    """Match Item 1/Business spans mentioning principal corporate offices are located in."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = (((span.get("structure") or {}).get("path_text")) or "")
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'item 1|business', path, re.I) and re.search(r'principal corporate offices', text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
