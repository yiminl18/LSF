def rule_page_range_1980s_ffo2(doc: dict) -> list[dict]:
    """Match likely answer tables on 1980s/early-1990s documents where FFO-2 often appears around pages 17-22."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            p = span.get("page_no")
            txt = (span.get("text") or "")
            if isinstance(p, int) and 17 <= p <= 22:
                if re.search(r'Individual|Withheld|Refunds|Net budget receipts', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
