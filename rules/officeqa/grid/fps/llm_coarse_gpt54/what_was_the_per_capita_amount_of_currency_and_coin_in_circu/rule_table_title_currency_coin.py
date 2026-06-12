def rule_table_title_currency_coin(doc: dict) -> list[dict]:
    """Match section headers or titles naming the currency/coin table directly."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if span.get("label") == "section_header" and re.search(r'currency\s+and\s+coin', txt, re.I):
                out.append(span)
            elif re.search(r'currency\s+and\s+coin\s+(outstanding\s+and\s+)?in\s+circulation', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
