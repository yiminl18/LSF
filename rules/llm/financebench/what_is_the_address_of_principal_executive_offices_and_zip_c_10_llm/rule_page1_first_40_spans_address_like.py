def rule_page1_first_40_spans_address_like(doc: dict) -> list[dict]:
    """Match address-like spans among the first 40 spans, where cover-page metadata usually appears."""
    try:
        import re
        spans = doc.get("texts", [])[:40]
        out = []
        for span in spans:
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            low = txt.lower()
            if "washington, d.c. 20549" in low:
                continue
            if re.search(r"\b\d{5}(?:-\d{4})?\b", txt) and re.search(r"\b(avenue|drive|road|plaza|street|way|boulevard|lane)\b", low):
                out.append(span)
            elif re.search(r"\b\d{1,6}\b", txt) and "address of principal executive offices" in low:
                out.append(span)
        return out
    except Exception:
        return []
