def rule_business_sentence_trades_on_exchange(doc: dict) -> list[dict]:
    """Match business-description sentences saying common stock trades/listed on an exchange under a symbol."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r"common stock.*?(trades|traded|listed).*?(nasdaq|new york stock exchange|nyse).*?symbol", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
