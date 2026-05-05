def rule_form_header_page1(doc: dict) -> list[dict]:
    """Retrieve the page-1 SEC form header span (10-K, 10-Q, or 8-K)."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            if span.get("bold") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if not txt:
                continue
            if re.fullmatch(r"FORM\s+(10-K|10-Q|8-K)", txt, flags=re.I):
                if span.get("label") in {"section_header", "text"}:
                    out.append(span)
        return out
    except Exception:
        return []

