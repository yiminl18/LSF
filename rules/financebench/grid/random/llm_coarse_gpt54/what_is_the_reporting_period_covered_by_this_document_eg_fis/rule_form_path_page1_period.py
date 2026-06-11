def rule_form_path_page1_period(doc: dict) -> list[dict]:
    """Match page-1 spans under a FORM 10-K/10-Q/8-K path that contain period keywords."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r'FORM\s+10-(K|Q|8-K)', path, re.I):
                if re.search(r'(ended|date of report|event reported)', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
