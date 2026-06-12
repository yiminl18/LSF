def rule_profile_economy_first_months_average(doc: dict) -> list[dict]:
    """Match spans in Profile of the Economy that state average monthly payroll growth over the first N months of the year."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if span.get("label") == "text":
                if (
                    "Profile of the Economy" in path
                    and re.search(r'first\s+\d+\s+months', txt, re.I)
                    and re.search(r'(nonfarm payroll|payroll job|job creation)', txt, re.I)
                    and re.search(r'(average|averaged)\s+\d[\d,]*', txt, re.I)
                    and re.search(r'(per month|jobs per month)', txt, re.I)
                ):
                    out.append(span)
        return out
    except Exception:
        return []
