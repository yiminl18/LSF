def rule_page1_address_line_preceded_by_ein_line(doc: dict) -> list[dict]:
    """Match address-looking spans immediately preceded by an EIN-looking span."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i in range(1, len(texts)):
            a, b = texts[i-1], texts[i]
            if a.get("page_no") != 1 or b.get("page_no") != 1:
                continue
            ta, tb = (a.get("text") or "").strip(), (b.get("text") or "").strip()
            if re.search(r'^\d{2,3}-\d{6,7}$|^\d{2}-\d{7}$', ta) and re.search(r'^\d{1,5}\s', tb):
                out.extend([a, b])
        return out
    except Exception:
        return []
