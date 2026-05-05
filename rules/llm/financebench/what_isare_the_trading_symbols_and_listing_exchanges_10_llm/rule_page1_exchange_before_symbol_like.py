def rule_page1_exchange_before_symbol_like(doc: dict) -> list[dict]:
    """Match symbol-like spans that appear shortly before/after exchange-name spans on page 1."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            t = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if span.get("page_no") == 1 and ("exchange" in t or "nasdaq" in t or "new york stock exchange" in t or "nyse" in t):
                for j in range(max(0, i - 6), min(len(texts), i + 3)):
                    s = texts[j]
                    txt = (s.get("text") or "").strip()
                    if s.get("page_no") == 1 and re.fullmatch(r"[A-Z]{1,6}[0-9A-Z]{0,4}", txt):
                        out.append(s)
        return out
    except Exception:
        return []
