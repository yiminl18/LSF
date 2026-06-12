def rule_contents_c2_per_capita(doc: dict) -> list[dict]:
    """Match contents-page spans mentioning C-2 or per capita comparative totals."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r'\bC-?2\b', txt, re.I) and re.search(r'per\s+capita', txt, re.I):
                out.append(span)
            elif re.search(r'per\s+capita\s+comparative\s+totals', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
