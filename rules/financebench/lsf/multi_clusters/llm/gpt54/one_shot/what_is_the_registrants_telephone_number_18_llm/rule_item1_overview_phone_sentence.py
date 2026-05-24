def rule_item1_overview_phone_sentence(doc: dict) -> list[dict]:
    """Match overview/business spans that include office address and telephone sentence."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = span.get("structure", {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if re.search(r"overview|item 1|business", path, re.I) and re.search(r"executive offices.*telephone number", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
