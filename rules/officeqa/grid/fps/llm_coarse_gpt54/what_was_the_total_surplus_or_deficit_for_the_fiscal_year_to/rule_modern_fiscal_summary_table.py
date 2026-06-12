def rule_modern_fiscal_summary_table(doc: dict) -> list[dict]:
    """Match modern Treasury Bulletin fiscal summary tables with on-budget/off-budget and fiscal year to date."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if (
                re.search(r'on-budget', txt, re.I)
                and re.search(r'off-budget', txt, re.I)
                and re.search(r'fiscal\s+\d{4}\s+to\s+date', txt, re.I)
            ):
                out.append(span)
    except Exception:
        return []
    return out
