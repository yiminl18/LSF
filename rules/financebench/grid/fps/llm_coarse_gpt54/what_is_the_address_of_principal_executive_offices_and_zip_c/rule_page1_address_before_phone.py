def rule_page1_address_before_phone(doc: dict) -> list[dict]:
    """Match address-like spans on page 1 that occur before the registrant telephone number block."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts[:-1]):
            if span.get("page_no") != 1:
                continue
            t = (span.get("text") or "").strip()
            if not (re.search(r'^\d{1,6}\s+\S+|\bone\b', t, re.I) and re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', t, re.I)):
                continue
            window = texts[i+1:i+5]
            joined = " ".join((w.get("text") or "") + " " + (w.get("text_span") or "") for w in window)
            if re.search(r"registrant.?s telephone number|telephone number, including area code|\(\d{3}\)", joined, re.I):
                out.append(span)
        return out
    except Exception:
        return []
