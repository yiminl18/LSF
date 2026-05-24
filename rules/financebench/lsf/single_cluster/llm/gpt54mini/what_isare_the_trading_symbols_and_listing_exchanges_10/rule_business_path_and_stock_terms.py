def rule_business_path_and_stock_terms(doc: dict) -> list[dict]:
    """Match spans under Item 1 / Business whose text mentions common stock, exchange, or symbol."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "")
            if re.search(r"item\s*1|business", path, re.I) and re.search(r"common stock|symbol|listed on|trades on|nasdaq|new york stock exchange|nyse", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
