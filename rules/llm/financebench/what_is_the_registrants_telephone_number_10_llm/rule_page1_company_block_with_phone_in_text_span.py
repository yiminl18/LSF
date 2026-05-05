def rule_page1_company_block_with_phone_in_text_span(doc: dict) -> list[dict]:
    """Match company-name cover headers whose text_span embeds the phone number."""
    import re
    try:
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b|\b\d{3}[-]\d{4}[-]\d{4}\b)")
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if span.get("label") != "section_header":
                continue
            if not span.get("text_span"):
                continue
            if phone_re.search(span.get("text_span") or "") and "exact name of registrant" in (span.get("text_span") or "").lower():
                out.append(span)
        return out
    except Exception:
        return []
