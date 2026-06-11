def rule_page1_phone_in_same_span_as_exact_name_block(doc: dict) -> list[dict]:
    """Match page-1 spans that combine exact-name block metadata and a phone number."""
    try:
        import re
        out = []
        phone_pat = re.compile(r"(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,12})")
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r"exact name of registrant", txt, re.I) and phone_pat.search(txt):
                out.append(span)
        return out
    except Exception:
        return []
