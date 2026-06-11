def rule_page1_phone_excluding_contacts_sections(doc: dict) -> list[dict]:
    """Match page-1 phone-like spans while excluding later contact lists and investor-relations sections."""
    try:
        import re
        out = []
        phone_pat = re.compile(r"(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,12})")
        for span in doc.get("texts", []):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            path = ((span.get("structure") or {}).get("path_text") or "")
            if span.get("page_no") == 1 and phone_pat.search(txt):
                if not re.search(r"contact|investor relations|press release", path + " " + txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
