def rule_fiscal_summary_table_with_other_means(doc: dict) -> list[dict]:
    """Match quarter-summary tables with an 'Other means' financing row."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "")
                if re.search(r'other means', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
