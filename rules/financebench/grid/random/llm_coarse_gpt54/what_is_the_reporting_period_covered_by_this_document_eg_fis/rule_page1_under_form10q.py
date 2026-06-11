def rule_page1_under_form10q(doc: dict) -> list[dict]:
    """Match page-1 spans under FORM 10-Q path that mention quarterly period ended."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r'FORM\s+10-Q', path, re.I):
                if re.search(r'quarterly period ended', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
