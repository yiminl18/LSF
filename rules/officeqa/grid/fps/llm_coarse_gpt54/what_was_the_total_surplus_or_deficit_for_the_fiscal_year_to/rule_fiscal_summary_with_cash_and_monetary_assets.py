def rule_fiscal_summary_with_cash_and_monetary_assets(doc: dict) -> list[dict]:
    """Match tables containing total surplus/deficit and cash/monetary assets columns."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'total surplus.*deficit', txt, re.I) and re.search(r'cash and monetary assets', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
