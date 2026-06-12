def rule_fiscal_summary_table_with_total_receipts_total_outlays(doc: dict) -> list[dict]:
    """Match summary tables with total receipts, total outlays, and total surplus/deficit."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if (
                re.search(r'total receipts', txt, re.I)
                and re.search(r'total outlays', txt, re.I)
                and re.search(r'total surplus.*deficit', txt, re.I)
            ):
                out.append(span)
    except Exception:
        return []
    return out
