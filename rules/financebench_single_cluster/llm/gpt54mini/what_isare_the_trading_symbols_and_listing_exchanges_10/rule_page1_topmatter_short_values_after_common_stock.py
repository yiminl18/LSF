def rule_page1_topmatter_short_values_after_common_stock(doc: dict) -> list[dict]:
    """Match short value spans following a 'Common Stock' span in page 1 top matter."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "")
            if span.get("page_no") == 1 and re.search(r"common stock", txt, re.I):
                for j in range(i + 1, min(len(texts), i + 6)):
                    s = texts[j]
                    t = (s.get("text") or "").strip()
                    if s.get("page_no") != 1:
                        continue
                    if re.fullmatch(r"[A-Z]{1,8}(?:\d+[A-Z]{0,3})?", t) or re.search(r"nasdaq|new york stock exchange|nyse", t, re.I):
                        out.append(s)
        return out
    except Exception:
        return []
