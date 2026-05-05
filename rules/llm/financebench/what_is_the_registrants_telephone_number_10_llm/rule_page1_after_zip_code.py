def rule_page1_after_zip_code(doc: dict) -> list[dict]:
    """Match page-1 spans where phone text appears near a zip-code label."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or ""))
            if span.get("page_no") == 1 and re.search(r"zip code.{0,120}(telephone|area code|\(\d{3}\))", text, re.I | re.S):
                out.append(span)
        return out
    except Exception:
        return []
