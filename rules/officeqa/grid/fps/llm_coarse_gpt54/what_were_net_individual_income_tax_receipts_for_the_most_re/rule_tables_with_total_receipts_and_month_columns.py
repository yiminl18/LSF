def rule_tables_with_total_receipts_and_month_columns(doc: dict) -> list[dict]:
    """Match quarter summary tables with Total receipts/Total budget receipts and month columns."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if span.get("label") == "table":
                if re.search(r'Total (?:budget )?receipts', txt, re.I) and (
                    re.search(r'July', txt, re.I) or re.search(r'October|Oct', txt, re.I)
                ):
                    out.append(span)
    except Exception:
        return []
    return out
