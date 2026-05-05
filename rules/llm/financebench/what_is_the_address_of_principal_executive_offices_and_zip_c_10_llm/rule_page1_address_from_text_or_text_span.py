def rule_page1_address_from_text_or_text_span(doc: dict) -> list[dict]:
    """Match spans where either text or text_span contains an address-like answer candidate."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for span in spans:
            if span.get("page_no") != 1:
                continue
            combined = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            low = combined.lower()
            if "washington, d.c. 20549" in low:
                continue
            if re.search(r"\b\d{1,6}\b", combined) and re.search(r"\b(avenue|drive|road|plaza|street|way)\b", low):
                out.append(span)
            elif re.search(r"\b\d{5}(?:-\d{4})?\b", combined) and "principal executive offices" in low:
                out.append(span)
        return out
    except Exception:
        return []
