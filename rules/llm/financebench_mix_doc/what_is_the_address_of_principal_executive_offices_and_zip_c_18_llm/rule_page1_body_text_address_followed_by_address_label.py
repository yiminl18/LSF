def rule_page1_body_text_address_followed_by_address_label(doc: dict) -> list[dict]:
    """Match page-1 body text spans where the next span is the address label."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts) - 1):
            a, b = texts[i], texts[i+1]
            if a.get("page_no") != 1 or b.get("page_no") != 1:
                continue
            ta, tb = (a.get("text") or "").strip(), (b.get("text") or "").strip()
            if re.search(r'^\d{1,5}\s', ta) and re.search(r'address of principal executive offices', tb, re.I):
                out.extend([a, b])
        return out
    except Exception:
        return []
