def rule_page1_address_with_office_label_in_same_or_next(doc: dict) -> list[dict]:
    """Match address-like spans whose same or next span contains the office-address label."""
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
            same = (span.get("text") or "") + " " + (span.get("text_span") or "")
            nxt = ""
            if i + 1 < len(texts):
                nxt = (texts[i+1].get("text") or "") + " " + (texts[i+1].get("text_span") or "")
            if re.search(r'address of principal executive offices', same + " " + nxt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
