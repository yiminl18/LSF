def rule_page1_address_before_zip_header(doc: dict) -> list[dict]:
    """Match address-like spans immediately preceding a zip-code-only header span."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts) - 1):
            a, b = texts[i], texts[i+1]
            if a.get("page_no") != 1 or b.get("page_no") != 1:
                continue
            ta, tb = (a.get("text") or "").strip(), (b.get("text") or "").strip()
            if re.search(r'^\d{1,5}\s', ta) and re.fullmatch(r'\d{5}(?:-\d{4})?', tb):
                out.append(a)
                out.append(b)
        return out
    except Exception:
        return []
