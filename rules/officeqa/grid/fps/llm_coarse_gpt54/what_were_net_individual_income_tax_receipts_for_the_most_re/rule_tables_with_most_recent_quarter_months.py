def rule_tables_with_most_recent_quarter_months(doc: dict) -> list[dict]:
    """Match FFO-2 tables that contain month columns/rows for quarter months like July/August/September or Oct/Nov/Dec."""
    import re
    out = []
    try:
        month_sets = [
            r'July.*August.*September',
            r'Oct(?:ober)?\..*Nov(?:ember)?\..*Dec(?:ember)?',
            r'Oct(?:ober)?.*Nov(?:ember)?.*Dec(?:ember)?',
        ]
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'Individual', txt, re.I):
                for pat in month_sets:
                    if re.search(pat, txt, re.I | re.S):
                        out.append(span)
                        break
    except Exception:
        return []
    return out
