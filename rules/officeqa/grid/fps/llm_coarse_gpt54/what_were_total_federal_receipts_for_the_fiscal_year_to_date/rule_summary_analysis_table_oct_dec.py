def rule_summary_analysis_table_oct_dec(doc: dict) -> list[dict]:
    """Match analysis summary tables with October-December and Actual fiscal year to date columns."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = span.get("text") or ""
                if re.search(r'October-December', txt, re.I) and re.search(r'Actual fiscal year to date', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
