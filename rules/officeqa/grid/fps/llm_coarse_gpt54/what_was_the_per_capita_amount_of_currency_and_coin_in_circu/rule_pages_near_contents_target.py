def rule_pages_near_contents_target(doc: dict) -> list[dict]:
    """Return spans on or near pages named in contents entries for currency/coin tables."""
    import re
    out = []
    try:
        target_pages = set()
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r'(MS-?1|USCC-?1|USCC-?2|C-?1|C-?2)', txt, re.I) and re.search(r'currency|coin|circulation|per\s+capita', txt, re.I):
                for n in re.findall(r'\b\d{1,4}\b', txt):
                    try:
                        target_pages.add(int(n))
                    except Exception:
                        pass
        if not target_pages:
            return []
        expanded = set()
        for p in target_pages:
            expanded.update({p - 1, p, p + 1})
        for span in doc.get("texts", []):
            if span.get("page_no") in expanded:
                out.append(span)
    except Exception:
        return []
    return out
