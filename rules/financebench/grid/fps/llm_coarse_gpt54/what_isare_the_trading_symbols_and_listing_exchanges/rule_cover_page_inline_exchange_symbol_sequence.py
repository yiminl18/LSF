def rule_cover_page_inline_exchange_symbol_sequence(doc: dict) -> list[dict]:
    """Match inline page-1 sequences where exchange and symbol appear in adjacent spans."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts) - 1):
            a, b = texts[i], texts[i + 1]
            if a.get("page_no") != 1 or b.get("page_no") != 1:
                continue
            ta, tb = (a.get("text") or "").strip(), (b.get("text") or "").strip()
            if re.search(r"NASDAQ|NYSE|New York Stock Exchange|Nasdaq", ta, re.I) and re.fullmatch(r"[A-Z0-9./-]{1,15}", tb):
                out.extend([a, b])
        return out
    except Exception:
        return []
