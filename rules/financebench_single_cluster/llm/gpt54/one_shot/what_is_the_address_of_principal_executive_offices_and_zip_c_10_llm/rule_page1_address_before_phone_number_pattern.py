def rule_page1_address_before_phone_number_pattern(doc: dict) -> list[dict]:
    """Match spans immediately before a page-1 phone-number-looking span."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for i in range(1, len(spans)):
            cur = spans[i]
            if cur.get("page_no") != 1:
                continue
            txt = (cur.get("text") or "").strip()
            if re.search(r"\(?\+?\d[\d ()-]{6,}\d", txt) or "telephone number" in txt.lower():
                prev = spans[i - 1]
                if prev.get("page_no") == 1:
                    out.append(prev)
        return out
    except Exception:
        return []
