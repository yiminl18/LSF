def rule_tables_with_recent_month_rows(doc: dict) -> list[dict]:
    """Match detailed receipt tables containing recent month rows and individual tax columns."""
    import re
    out = []
    try:
        recent_months = [r'Jan', r'Feb', r'Mar', r'Apr', r'May', r'June', r'July', r'Aug', r'Sept', r'Oct', r'Nov', r'Dec']
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'Individual', txt, re.I):
                hits = sum(1 for m in recent_months if re.search(m, txt, re.I))
                if hits >= 4:
                    out.append(span)
    except Exception:
        return []
    return out
