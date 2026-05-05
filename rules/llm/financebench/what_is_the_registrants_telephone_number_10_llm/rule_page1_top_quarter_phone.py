def rule_page1_top_quarter_phone(doc: dict) -> list[dict]:
    """Match phone-like spans in the first quarter of page-1 span order."""
    import re
    try:
        texts = doc.get("texts", [])
        page1 = [s for s in texts if s.get("page_no") == 1]
        cutoff = max(1, len(page1) // 4)
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b|\b\d{3}[-]\d{4}[-]\d{4}\b)")
        out = []
        for span in page1[:cutoff]:
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if phone_re.search(text):
                out.append(span)
        return out
    except Exception:
        return []
