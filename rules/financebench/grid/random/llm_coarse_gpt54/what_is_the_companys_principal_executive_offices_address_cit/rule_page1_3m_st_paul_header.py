def rule_page1_3m_st_paul_header(doc: dict) -> list[dict]:
    """Match 3M-style page-1 address header with city/state but no zip in same span."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.search(r'3M Center,\s*St\. Paul,\s*Minnesota', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
