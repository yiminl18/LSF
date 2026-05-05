def rule_page1_exchange_after_symbol_like(doc: dict) -> list[dict]:
    """Match exchange-name spans that appear shortly after a symbol-like span on page 1."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.fullmatch(r"[A-Z]{1,6}[0-9A-Z]{0,4}", txt):
                for j in range(i + 1, min(len(texts), i + 6)):
                    s = texts[j]
                    t = ((s.get("text") or "") + " " + (s.get("text_span") or "")).lower()
                    if s.get("page_no") == 1 and ("exchange" in t or "nasdaq" in t or "new york stock exchange" in t or "nyse" in t):
                        out.append(s)
        return out
    except Exception:
        return []
