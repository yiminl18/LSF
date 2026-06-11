def rule_page1_symbol_like_spans(doc: dict) -> list[dict]:
    """Match page-1 short ticker-like bold spans that often hold the trading symbol."""
    import re
    try:
        out = []
        for s in doc.get("texts", []):
            txt = (s.get("text") or "").strip()
            if s.get("page_no") != 1:
                continue
            if len(txt) <= 15 and re.fullmatch(r"[A-Z0-9./-]{1,15}", txt):
                if s.get("bold") == 1 or s.get("all_cap") == 1:
                    out.append(s)
        return out
    except Exception:
        return []
