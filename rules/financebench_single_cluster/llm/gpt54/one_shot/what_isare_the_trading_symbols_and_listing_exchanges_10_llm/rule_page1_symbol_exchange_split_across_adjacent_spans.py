def rule_page1_symbol_exchange_split_across_adjacent_spans(doc: dict) -> list[dict]:
    """Match adjacent page 1 spans where one looks like a symbol and the next looks like an exchange."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts) - 1):
            a = texts[i]
            b = texts[i + 1]
            ta = (a.get("text") or "").strip()
            tb = (b.get("text") or "").strip()
            if a.get("page_no") == 1 and b.get("page_no") == 1:
                if re.fullmatch(r"[A-Z]{1,8}(?:\d+[A-Z]{0,3})?", ta) and re.search(r"nasdaq|new york stock exchange|nyse", tb, re.I):
                    out.extend([a, b])
        return out
    except Exception:
        return []
