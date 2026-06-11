def rule_cover_page_company_block_phone(doc: dict) -> list[dict]:
    """Match phone-number spans in the cover-page company block before Part I starts."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        part_i_idx = None
        for i, span in enumerate(texts):
            if re.search(r"\bPART I\b", span.get("text", "") or "", re.I):
                part_i_idx = i
                break
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})")
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            if part_i_idx is not None and i >= part_i_idx:
                continue
            if phone_re.search(span.get("text", "") or ""):
                out.append(span)
        return out
    except Exception:
        return []
