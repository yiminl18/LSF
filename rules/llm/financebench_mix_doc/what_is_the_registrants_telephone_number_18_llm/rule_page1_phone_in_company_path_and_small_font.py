def rule_page1_phone_in_company_path_and_small_font(doc: dict) -> list[dict]:
    """Match small-font page-1 company-block spans containing phone clues."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if (span.get("size") or 99) > 10:
                continue
            path = span.get("structure", {}).get("path_text", "") or ""
            if re.search(r"form 10-|current report|commission", path, re.I):
                continue
            blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
            if re.search(r"telephone number|area code", blob, re.I) or re.search(r"(?:\+?\d{1,3}\s*)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}", blob):
                out.append(span)
        return out
    except Exception:
        return []
