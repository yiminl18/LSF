def rule_total_surplus_and_fiscal_to_date_same_table(doc: dict) -> list[dict]:
    """Match tables that contain both 'Total surplus or deficit' and a fiscal-to-date row."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            has_col = re.search(r'total surplus or deficit', txt, re.I) is not None
            has_row = re.search(r'fiscal\s+(year|\d{4})\s+to\s+date', txt, re.I) is not None
            if has_col and has_row:
                out.append(span)
    except Exception:
        return []
    return out
