def rule_fd1_summary_of_federal_debt_tables(doc: dict) -> list[dict]:
    """Match tables under or mentioning FD-1 / Summary of Federal Debt."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if (
                re.search(r'fd[\-–— ]?1', path, re.I)
                or re.search(r'summary of federal debt', path, re.I)
                or re.search(r'fd[\-–— ]?1', text, re.I)
                or re.search(r'summary of federal debt', text, re.I)
            ):
                out.append(span)
    except Exception:
        return []
    return out
