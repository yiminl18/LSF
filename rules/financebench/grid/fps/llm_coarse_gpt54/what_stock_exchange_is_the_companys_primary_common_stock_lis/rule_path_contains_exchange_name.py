def rule_path_contains_exchange_name(doc: dict) -> list[dict]:
    """Match spans whose structure.path_text itself contains the exchange name."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure", {}).get("path_text") or "").strip()
            if re.search(r'new york stock exchange|the nasdaq global select market|the nasdaq stock market llc|NASDAQ\b', path, re.I):
                out.append(span)
        return out
    except Exception:
        return []
