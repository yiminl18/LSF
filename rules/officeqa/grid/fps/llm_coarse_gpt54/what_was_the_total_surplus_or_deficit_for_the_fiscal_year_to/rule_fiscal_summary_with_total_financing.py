def rule_fiscal_summary_with_total_financing(doc: dict) -> list[dict]:
    """Match tables where total financing appears alongside fiscal surplus/deficit."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'total financing', txt, re.I) and re.search(r'(surplus|deficit)', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
