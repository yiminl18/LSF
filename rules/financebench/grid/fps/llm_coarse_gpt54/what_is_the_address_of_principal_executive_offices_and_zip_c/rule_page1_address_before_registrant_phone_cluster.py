def rule_page1_address_before_registrant_phone_cluster(doc: dict) -> list[dict]:
    """Match address-like spans in the cover-page cluster before the registrant phone line."""
    try:
        import re
        texts = [s for s in doc.get("texts", []) if s.get("page_no") == 1]
        phone_idx = None
        for i, s in enumerate(texts):
            full = (s.get("text") or "") + " " + (s.get("text_span") or "")
            if re.search(r"registrant.?s telephone number|telephone number, including area code", full, re.I):
                phone_idx = i
                break
        if phone_idx is None:
            return []
        out = []
        for s in texts[max(0, phone_idx-6):phone_idx]:
            t = (s.get("text") or "").strip()
            if re.search(r'^\d{1,6}\s+\S+|\bone\b', t, re.I) and re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', t, re.I):
                out.append(s)
        return out
    except Exception:
        return []
