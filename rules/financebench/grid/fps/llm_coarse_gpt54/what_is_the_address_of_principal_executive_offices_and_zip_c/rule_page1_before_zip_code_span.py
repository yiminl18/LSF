def rule_page1_before_zip_code_span(doc: dict) -> list[dict]:
    """Match address-like spans on page 1 that occur immediately before a zip-code-labeled span."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts[:-1]):
            if span.get("page_no") != 1:
                continue
            t = span.get("text") or ""
            nxt = texts[i+1]
            nxt_full = (nxt.get("text") or "") + " " + (nxt.get("text_span") or "")
            if re.search(r'^\d{1,6}\s+\S+|\bone\b', t, re.I) and re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', t, re.I):
                if re.search(r'zip code', nxt_full, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
