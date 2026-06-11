def rule_page1_h2_or_h3_address_heading(doc: dict) -> list[dict]:
    """Match page-1 H2/H3 address headings in the registrant information block."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            level = ((span.get("structure") or {}).get("level") or "")
            if level not in {"H2", "H3"}:
                continue
            t = (span.get("text") or "").strip()
            if re.search(r'^\d{1,6}\s+\S+|\bone\b', t, re.I) and re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', t, re.I):
                out.append(span)
        return out
    except Exception:
        return []
