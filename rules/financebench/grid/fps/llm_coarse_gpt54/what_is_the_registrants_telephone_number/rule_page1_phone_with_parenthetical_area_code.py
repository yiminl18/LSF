def rule_page1_phone_with_parenthetical_area_code(doc: dict) -> list[dict]:
    """Match page-1 spans with a parenthetical area code phone format."""
    try:
        import re
        out = []
        pat = re.compile(r"\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}")
        for span in doc.get("texts", []):
            if span.get("page_no") == 1:
                txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
                if pat.search(txt):
                    out.append(span)
        return out
    except Exception:
        return []
