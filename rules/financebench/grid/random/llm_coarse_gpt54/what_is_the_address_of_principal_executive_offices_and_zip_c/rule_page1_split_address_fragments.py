def rule_page1_split_address_fragments(doc: dict) -> list[dict]:
    """Match consecutive page-1 spans that together form a split address block."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts) - 1):
            a, b = texts[i], texts[i + 1]
            if a.get("page_no") == 1 and b.get("page_no") == 1:
                ta = a.get("text") or ""
                tb = b.get("text") or ""
                if re.search(r'^\d{2,}.*(?:Street|Avenue|Boulevard|Drive|Road|Center|Plaza)', ta, re.I) and re.search(r'^[A-Z][a-zA-Z .,-]+$', tb):
                    out.extend([a, b])
        return out
    except Exception:
        return []
