def rule_exchange_in_same_span_as_symbol_and_class(doc: dict) -> list[dict]:
    """Match spans whose combined text mentions class, symbol, and exchange together."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            combined = " ".join([span.get("text", "") or "", span.get("text_span", "") or ""])
            if re.search(r'common stock', combined, re.I) and re.search(r'(trading symbol|symbol)', combined, re.I) and re.search(r'new york stock exchange|nasdaq|global select market', combined, re.I):
                out.append(span)
        return out
    except Exception:
        return []
