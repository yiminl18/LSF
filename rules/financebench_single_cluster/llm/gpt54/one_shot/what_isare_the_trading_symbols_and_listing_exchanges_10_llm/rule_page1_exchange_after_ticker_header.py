def rule_page1_exchange_after_ticker_header(doc: dict) -> list[dict]:
    """Match exchange span immediately after a ticker-like section header on page 1."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and span.get("label") == "section_header" and re.fullmatch(r"[A-Z]{2,8}(?:\d+[A-Z]{0,3})?", txt):
                for j in range(i + 1, min(len(texts), i + 4)):
                    s = texts[j]
                    t = (s.get("text") or "")
                    if re.search(r"nasdaq|new york stock exchange|nyse", t, re.I):
                        out.append(s)
        return out
    except Exception:
        return []
