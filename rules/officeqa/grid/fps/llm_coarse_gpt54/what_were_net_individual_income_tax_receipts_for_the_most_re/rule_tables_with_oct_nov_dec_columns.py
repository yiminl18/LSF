def rule_tables_with_oct_nov_dec_columns(doc: dict) -> list[dict]:
    """Match quarter tables with October/November/December columns, often directly containing the answer month value."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if span.get("label") == "table":
                if re.search(r'Oct|October', txt, re.I) and re.search(r'Nov|November', txt, re.I) and re.search(r'Dec|December', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
