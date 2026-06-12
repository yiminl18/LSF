def rule_any_span_with_currency_coin_and_circulation(doc: dict) -> list[dict]:
    """Match any span mentioning currency, coin, and circulation together."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r'currency', txt, re.I) and re.search(r'coin', txt, re.I) and re.search(r'circulation', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
