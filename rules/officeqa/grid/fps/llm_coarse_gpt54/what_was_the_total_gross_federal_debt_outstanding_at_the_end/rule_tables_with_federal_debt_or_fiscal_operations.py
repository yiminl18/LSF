def rule_tables_with_federal_debt_or_fiscal_operations(doc: dict) -> list[dict]:
    """Broad recall rule: match tables under either Federal Debt or Summary of Fiscal Operations paths."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if re.search(r'federal debt|summary of fiscal operations|ffo[\-–— ]?1|fd[\-–— ]?1', path, re.I):
                out.append(span)
    except Exception:
        return []
    return out
