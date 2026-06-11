def rule_page1_split_address_then_zip(doc: dict) -> list[dict]:
    """Match page-1 address spans when the ZIP code is split into a nearby following span."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            t = (span.get("text") or "").strip()
            if not (re.search(r'^\d{1,6}\s+\S+|\bone\b', t, re.I) and re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', t, re.I)):
                continue
            window = texts[i+1:i+4]
            joined = " ".join((w.get("text") or "") + " " + (w.get("text_span") or "") for w in window)
            if re.search(r'\b\d{5}(?:-\d{4})?\b', joined) or re.search(r'zip code', joined, re.I):
                out.append(span)
        return out
    except Exception:
        return []
