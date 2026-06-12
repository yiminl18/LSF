def rule_tables_with_july_august_september_columns(doc: dict) -> list[dict]:
    """Match quarter tables with July/August/September columns, often directly containing the answer month value."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if span.get("label") == "table":
                if re.search(r'July', txt, re.I) and re.search(r'August', txt, re.I) and re.search(r'September', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
