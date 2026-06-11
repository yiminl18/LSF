def rule_page1_near_exact_name_of_registrant(doc: dict) -> list[dict]:
    """Match page-1 spans near the registrant name block where the address usually appears."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            neighborhood = texts[max(0, i-4): min(len(texts), i+5)]
            joined = " ".join(((s.get("text") or "") + " " + (s.get("text_span") or "")) for s in neighborhood)
            t = span.get("text") or ""
            if re.search(r'exact name of registrant', joined, re.I):
                if re.search(r'^\d{1,6}\s+\S+|\bone\b', t, re.I) and re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', t, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
