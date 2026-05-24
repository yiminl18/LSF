def rule_page1_two_column_label_number_pattern(doc: dict) -> list[dict]:
    """Match numeric spans in cover-page two-column layouts where the prior span is an outstanding-share label."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i in range(1, len(texts)):
            prev = texts[i - 1]
            cur = texts[i]
            if cur.get("page_no") not in (1, 2):
                continue
            if prev.get("page_no") != cur.get("page_no"):
                continue
            if re.fullmatch(r"[\d,]{6,}", (cur.get("text") or "").strip()):
                p = (prev.get("text") or "").lower()
                if "number of shares of common stock outstanding" in p or "shares of common stock outstanding" in p:
                    out.append(cur)
        return out
    except Exception:
        return []
