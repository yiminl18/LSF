def rule_page1_address_with_zipcode_parenthetical_same_cluster(doc: dict) -> list[dict]:
    """Match address spans in clusters where '(Zip Code)' appears in the same or adjacent span."""
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
            cluster = texts[max(0, i-1): min(len(texts), i+3)]
            joined = " ".join((c.get("text") or "") + " " + (c.get("text_span") or "") for c in cluster)
            if re.search(r'zip code', joined, re.I):
                out.append(span)
        return out
    except Exception:
        return []
