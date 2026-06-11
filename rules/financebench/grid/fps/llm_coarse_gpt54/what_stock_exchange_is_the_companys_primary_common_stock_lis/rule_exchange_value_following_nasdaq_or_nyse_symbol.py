def rule_exchange_value_following_nasdaq_or_nyse_symbol(doc: dict) -> list[dict]:
    """Match exchange spans near stock ticker symbols on the cover page."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, s in enumerate(texts):
            t = (s.get("text") or "").strip()
            if s.get("page_no") == 1 and re.fullmatch(r'[A-Z]{2,5}', t or ""):
                for j in range(i, min(len(texts), i + 6)):
                    s2 = texts[j]
                    if s2.get("page_no") == 1 and re.search(r'new york stock exchange|nasdaq|global select market', s2.get("text", "") or "", re.I):
                        out.append(s2)
        return out
    except Exception:
        return []
