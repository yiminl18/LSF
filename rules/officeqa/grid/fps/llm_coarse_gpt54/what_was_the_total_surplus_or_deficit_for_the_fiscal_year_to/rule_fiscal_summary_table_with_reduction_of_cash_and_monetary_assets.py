def rule_fiscal_summary_table_with_reduction_of_cash_and_monetary_assets(doc: dict) -> list[dict]:
    """Match quarter-summary tables with a row 'Reduction of cash and monetary assets'."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "")
                if re.search(r'reduction of cash and monetary assets', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
