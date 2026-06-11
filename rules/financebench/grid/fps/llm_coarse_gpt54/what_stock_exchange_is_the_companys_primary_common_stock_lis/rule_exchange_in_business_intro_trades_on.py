def rule_exchange_in_business_intro_trades_on(doc: dict) -> list[dict]:
    """Match business intro spans saying the stock 'trades on' an exchange."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            combined = " ".join([span.get("text", "") or "", span.get("text_span", "") or ""])
            if re.search(r'(common stock|our stock).*trades on .*?(new york stock exchange|nasdaq)', combined, re.I):
                out.append(span)
        return out
    except Exception:
        return []
