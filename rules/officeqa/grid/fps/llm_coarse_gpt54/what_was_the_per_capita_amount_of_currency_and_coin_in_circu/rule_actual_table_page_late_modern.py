def rule_actual_table_page_late_modern(doc: dict) -> list[dict]:
    """Match later-document actual table pages where USCC tables often appear around pages 40-90+."""
    import re
    out = []
    try:
        candidate_pages = set()
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r'USCC-?1|USCC-?2|C-?1|C-?2', txt, re.I):
                for n in re.findall(r'\b\d{1,4}\b', txt):
                    try:
                        val = int(n)
                        if val >= 30:
                            candidate_pages.add(val)
                    except Exception:
                        pass
        for span in doc.get("texts", []):
            if span.get("page_no") in candidate_pages:
                out.append(span)
    except Exception:
        return []
    return out
