def rule_page1_textspan_contains_symbol_exchange(doc: dict) -> list[dict]:
    """Match page-1 spans whose text_span contains both a ticker-like token and an exchange name."""
    import re
    try:
        out = []
        for s in doc.get("texts", []):
            if s.get("page_no") != 1:
                continue
            ts = (s.get("text_span") or "")
            if re.search(r"NASDAQ|NYSE|New York Stock Exchange|Nasdaq", ts, re.I) and re.search(r"\b[A-Z]{1,6}(?:\d+[A-Z]*)?\b", ts):
                out.append(s)
        return out
    except Exception:
        return []
