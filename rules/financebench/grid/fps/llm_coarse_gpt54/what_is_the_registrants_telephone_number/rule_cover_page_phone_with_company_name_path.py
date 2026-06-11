def rule_cover_page_phone_with_company_name_path(doc: dict) -> list[dict]:
    """Match phone-like spans on page 1 whose path_text is under the company name rather than later sections."""
    try:
        import re
        out = []
        phone_pat = re.compile(r"(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,12})")
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            path = ((span.get("structure") or {}).get("path_text") or "")
            if phone_pat.search(txt) and path and not re.search(r"signatures|item\s+\d|part\s+[ivx]+", path, re.I):
                out.append(span)
        return out
    except Exception:
        return []
