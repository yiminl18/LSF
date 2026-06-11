def rule_exchange_in_first_40_spans(doc: dict) -> list[dict]:
    """Match exchange-name spans appearing very early in the document, usually on the cover page."""
    import re
    try:
        out = []
        for span in (doc.get("texts", []) or [])[:40]:
            txt = " ".join([span.get("text", "") or "", span.get("text_span", "") or ""])
            if re.search(r'new york stock exchange|the nasdaq global select market|the nasdaq stock market llc|NASDAQ\b', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
