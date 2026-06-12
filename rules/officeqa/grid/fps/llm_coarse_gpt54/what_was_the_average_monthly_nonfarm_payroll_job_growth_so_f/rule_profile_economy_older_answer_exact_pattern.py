def rule_profile_economy_older_answer_exact_pattern(doc: dict) -> list[dict]:
    """Match older answer pattern 'nonfarm payrolls averaged NNN,NNN per month over the first N months of this year'."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            if (
                span.get("label") == "text"
                and re.search(r'nonfarm payrolls averaged\s+\d[\d,]*\s+per month', txt, re.I)
                and re.search(r'first\s+\d+\s+months of this year', txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
