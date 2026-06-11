def rule_exchange_short_name_only(doc: dict) -> list[dict]:
    """Match short-form exchange-only spans like 'NASDAQ', 'NYSE', or 'New York Stock Exchange'."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if re.fullmatch(r'(NASDAQ|NYSE|New York Stock Exchange|The Nasdaq Global Select Market|The Nasdaq Stock Market LLC)', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
