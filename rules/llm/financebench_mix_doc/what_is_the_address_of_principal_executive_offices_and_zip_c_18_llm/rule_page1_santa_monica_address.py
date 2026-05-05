def rule_page1_santa_monica_address(doc: dict) -> list[dict]:
    """Match page-1 spans containing the Santa Monica address pattern."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r'3100 Ocean Park Boulevard.*Santa Monica.*90405', text, re.I) or re.search(r'2701 Olympic Boulevard.*Santa Monica.*90404', text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
