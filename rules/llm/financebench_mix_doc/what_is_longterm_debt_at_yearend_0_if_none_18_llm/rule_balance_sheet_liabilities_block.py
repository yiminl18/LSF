def rule_balance_sheet_liabilities_block(doc: dict) -> list[dict]:
    """Match balance sheet tables containing liabilities and debt together."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if ("liabilities" in txt and "debt" in txt) and ("balance sheet" in txt or "stockholders' equity" in txt or "shareholders' equity" in txt):
                out.append(span)
    except Exception:
        return []
    return out
