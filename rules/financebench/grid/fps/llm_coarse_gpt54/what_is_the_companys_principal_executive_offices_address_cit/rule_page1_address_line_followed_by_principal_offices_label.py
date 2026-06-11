def rule_page1_address_line_followed_by_principal_offices_label(doc: dict) -> list[dict]:
    """Match a page-1 address span immediately followed by a body span saying address of principal executive offices."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts[:-1]):
            nxt = texts[i + 1]
            if span.get("page_no") != 1 or nxt.get("page_no") != 1:
                continue
            text = (span.get("text") or "").strip()
            nxt_text = (nxt.get("text") or "").strip()
            if re.search(r'\d{1,6}\s+\S+', text) and (
                re.search(r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b', text) or
                re.search(r'\bUnited Kingdom\b|\bNew York\b|\bCalifornia\b|\bVirginia\b', text)
            ):
                if re.search(r'address of principal executive offices', nxt_text, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
