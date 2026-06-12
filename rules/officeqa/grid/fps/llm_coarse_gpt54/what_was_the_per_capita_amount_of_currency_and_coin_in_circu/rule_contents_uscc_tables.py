def rule_contents_uscc_tables(doc: dict) -> list[dict]:
    """Match contents-page spans mentioning USCC-1 or USCC-2 tables."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r'\bUSCC-?1\b', txt, re.I) or re.search(r'\bUSCC-?2\b', txt, re.I):
                out.append(span)
            elif re.search(r'amounts\s+outstanding\s+and\s+in\s+circulation', txt, re.I):
                out.append(span)
            elif re.search(r'per\s+capita\s+comparative\s+totals', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
