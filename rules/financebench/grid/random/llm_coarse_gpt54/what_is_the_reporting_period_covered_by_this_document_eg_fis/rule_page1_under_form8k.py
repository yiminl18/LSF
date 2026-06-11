def rule_page1_under_form8k(doc: dict) -> list[dict]:
    """Match page-1 spans under FORM 8-K path that mention date of report or event date."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r'FORM\s+8-K', path, re.I):
                if re.search(r'(Date of Report|event reported)', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
