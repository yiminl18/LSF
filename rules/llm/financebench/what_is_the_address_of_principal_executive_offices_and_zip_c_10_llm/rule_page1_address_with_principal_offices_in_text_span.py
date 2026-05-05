def rule_page1_address_with_principal_offices_in_text_span(doc: dict) -> list[dict]:
    """Match spans whose text is address-like and whose text_span names principal executive offices."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for span in spans:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            tsp = (span.get("text_span") or "").lower()
            if "principal executive offices" not in tsp:
                continue
            if re.search(r"\b\d{1,6}\b", txt) or re.search(r"\b\d{5}(?:-\d{4})?\b", txt):
                out.append(span)
        return out
    except Exception:
        return []
