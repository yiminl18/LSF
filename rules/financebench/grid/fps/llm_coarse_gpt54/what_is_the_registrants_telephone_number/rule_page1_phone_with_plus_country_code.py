def rule_page1_phone_with_plus_country_code(doc: dict) -> list[dict]:
    """Match page-1 spans with an international +country-code phone format."""
    try:
        import re
        out = []
        pat = re.compile(r"\+\d[\d\-\s]{6,}\d")
        for span in doc.get("texts", []):
            if span.get("page_no") == 1:
                txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
                if pat.search(txt):
                    out.append(span)
        return out
    except Exception:
        return []
