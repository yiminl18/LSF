def rule_page1_near_trading_symbol_label(doc: dict) -> list[dict]:
    """Match spans within a small window around a page 1 'Trading Symbol' label."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "")
            if span.get("page_no") == 1 and re.search(r"trading symbol", txt, re.I):
                for j in range(max(0, i - 3), min(len(texts), i + 6)):
                    s = texts[j]
                    t = (s.get("text") or "")
                    if re.search(r"trading symbol|exchange|registered", t, re.I) or re.fullmatch(r"[A-Z]{1,8}(?:\d+[A-Z]{0,3})?", t.strip()):
                        out.append(s)
        return out
    except Exception:
        return []
