def rule_summary_analysis_table_total_receipts(doc: dict) -> list[dict]:
    """Match analysis summary tables containing a Total receipts row."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = span.get("text") or ""
                if re.search(r'Total receipts', txt, re.I) and (
                    re.search(r'Budget estimates', txt, re.I)
                    or re.search(r'Actual fiscal year to date', txt, re.I)
                    or re.search(r'October-December', txt, re.I)
                ):
                    out.append(span)
        return out
    except Exception:
        return []
