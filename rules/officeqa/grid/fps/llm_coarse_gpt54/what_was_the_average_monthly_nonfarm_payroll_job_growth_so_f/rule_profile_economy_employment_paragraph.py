def rule_profile_economy_employment_paragraph(doc: dict) -> list[dict]:
    """Match text spans in Profile of the Economy discussing average nonfarm payroll job growth this year."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if span.get("label") == "text":
                if (
                    "Profile of the Economy" in path
                    and re.search(r'nonfarm payroll', txt, re.I)
                    and re.search(r'average(?:d)?\s+\d[\d,]*\s+(?:per month|jobs per month)', txt, re.I)
                ):
                    out.append(span)
        return out
    except Exception:
        return []
