def rule_page1_phone_followed_by_section12b_neighbor(doc: dict) -> list[dict]:
    """Match spans with phone numbers whose nearby neighbors mention Section 12(b)."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b|\b\d{3}[-]\d{4}[-]\d{4}\b)")
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if not phone_re.search(text):
                continue
            neigh = texts[i+1:i+4]
            neigh_text = " ".join(((n.get("text") or "") + " " + (n.get("text_span") or "")) for n in neigh).lower()
            if "section 12(b)" in neigh_text or "securities registered pursuant to section 12(b)" in neigh_text:
                out.append(span)
        return out
    except Exception:
        return []
