def rule_page1_address_line_followed_by_phone_line(doc: dict) -> list[dict]:
    """Match address-looking spans immediately followed by a phone-looking span."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts) - 1):
            a, b = texts[i], texts[i+1]
            if a.get("page_no") != 1 or b.get("page_no") != 1:
                continue
            ta, tb = (a.get("text") or "").strip(), (b.get("text") or "").strip()
            if re.search(r'^\d{1,5}\s', ta) and re.search(r'\(?\+?\d[\d\-\)\( ]{6,}', tb):
                out.extend([a, b])
        return out
    except Exception:
        return []
