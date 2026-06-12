def rule_contents_ms1_currency_and_coin(doc: dict) -> list[dict]:
    """Match contents-page spans mentioning MS-1 Currency and Coin in Circulation."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r'\bMS-?1\b', txt, re.I) and re.search(r'currency\s+and\s+coin\s+in\s+circulation', txt, re.I):
                out.append(span)
            elif re.search(r'CONTENTS', path, re.I) and re.search(r'currency\s+and\s+coin\s+in\s+circulation', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
