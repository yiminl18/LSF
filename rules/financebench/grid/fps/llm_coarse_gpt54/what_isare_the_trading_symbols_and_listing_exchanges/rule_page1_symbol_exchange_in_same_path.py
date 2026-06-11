def rule_page1_symbol_exchange_in_same_path(doc: dict) -> list[dict]:
    """Match page-1 spans whose path_text itself contains a symbol/exchange pair."""
    import re
    try:
        out = []
        for s in doc.get("texts", []):
            if s.get("page_no") != 1:
                continue
            path = ((s.get("structure") or {}).get("path_text") or "")
            if re.search(r"NASDAQ|NYSE|New York Stock Exchange|Nasdaq", path, re.I) and re.search(r"\b[A-Z]{1,6}(?:\d+[A-Z]*)?\b", path):
                out.append(s)
        return out
    except Exception:
        return []
