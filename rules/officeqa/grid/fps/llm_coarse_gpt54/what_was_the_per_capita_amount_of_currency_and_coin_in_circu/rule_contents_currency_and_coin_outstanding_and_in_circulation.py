def rule_contents_currency_and_coin_outstanding_and_in_circulation(doc: dict) -> list[dict]:
    """Match contents-page spans mentioning U.S. Currency and Coin Outstanding and in Circulation."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r'currency\s+and\s+coin\s+outstanding\s+and\s+in\s+circulation', txt, re.I):
                out.append(span)
            elif re.search(r'CONTENTS', path, re.I) and re.search(r'currency\s+and\s+coin', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
