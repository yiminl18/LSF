def rule_under_special_reports_currency_coin(doc: dict) -> list[dict]:
    """Match spans under Special Reports/Trust Funds mentioning currency and coin outstanding and in circulation."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "")
            if re.search(r'(special\s+reports|trust\s+funds?)', path, re.I) and re.search(r'currency\s+and\s+coin', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
