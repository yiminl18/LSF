def rule_federal_debt_section_tables(doc: dict) -> list[dict]:
    """Match all table spans under a Federal Debt section."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if re.search(r'federal debt', path, re.I):
                out.append(span)
    except Exception:
        return []
    return out
