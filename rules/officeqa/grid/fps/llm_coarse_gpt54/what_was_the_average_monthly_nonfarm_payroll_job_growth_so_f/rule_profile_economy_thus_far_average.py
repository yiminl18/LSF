def rule_profile_economy_thus_far_average(doc: dict) -> list[dict]:
    """Match spans in Profile of the Economy that say payroll/job growth has averaged X thus far this year."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if span.get("label") == "text":
                if (
                    "Profile of the Economy" in path
                    and re.search(r'thus far', txt, re.I)
                    and re.search(r'(job growth|payroll job creation|nonfarm payroll)', txt, re.I)
                    and re.search(r'(average|averaged)\s+\d[\d,]*', txt, re.I)
                ):
                    out.append(span)
        return out
    except Exception:
        return []
