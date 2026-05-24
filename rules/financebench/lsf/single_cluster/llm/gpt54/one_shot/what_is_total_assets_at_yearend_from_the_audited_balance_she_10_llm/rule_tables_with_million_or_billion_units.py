def rule_tables_with_million_or_billion_units(doc: dict) -> list[dict]:
    """Match financial statement tables that specify units like million or billion and include total assets."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "").lower()
            if "total assets" in text and ("million" in text or "billions" in text or "billion" in text):
                out.append(span)
        return out
    except Exception:
        return []
