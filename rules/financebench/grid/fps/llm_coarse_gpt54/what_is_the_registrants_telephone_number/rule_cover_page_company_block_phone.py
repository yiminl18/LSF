def rule_cover_page_company_block_phone(doc: dict) -> list[dict]:
    """Match phone-like spans under the main company cover-page block on page 1."""
    try:
        import re
        out = []
        phone_pat = re.compile(r"(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,12})")
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if phone_pat.search(txt) and path and not re.search(r"part i|item \d", path, re.I):
                out.append(span)
        return out
    except Exception:
        return []
