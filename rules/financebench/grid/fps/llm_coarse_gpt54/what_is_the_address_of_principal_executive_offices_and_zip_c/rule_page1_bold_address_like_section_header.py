def rule_page1_bold_address_like_section_header(doc: dict) -> list[dict]:
    """Match bold page-1 section headers that look like a street address and often carry the office-address label."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if span.get("bold") != 1:
                continue
            if span.get("label") not in {"section_header", "text"}:
                continue
            t = (span.get("text") or "").strip()
            ts = (span.get("text_span") or "").strip()
            full = (t + " " + ts).strip()
            if re.search(r'address of principal executive offices', full, re.I):
                out.append(span)
                continue
            if re.search(r'^\d{1,6}\s+\S+', t) or re.search(r'\b(one|one riverfront|one apple park)\b', t, re.I):
                if re.search(r'street|st\.?|avenue|ave\.?|drive|dr\.?|plaza|road|rd\.?|market|hamilton|rockledge|tower|water|lake|long bridge|penn', t, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
