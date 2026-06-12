def rule_page_with_currency_coin_table(doc: dict) -> list[dict]:
    """Match table spans whose text suggests the currency/coin table itself."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'currency\s+and\s+coin', txt, re.I):
                out.append(span)
            elif re.search(r'amounts\s+outstanding\s+and\s+in\s+circulation', txt, re.I):
                out.append(span)
            elif re.search(r'per\s+capita', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
