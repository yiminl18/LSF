def rule_tables_with_total_assets_and_parentheses_negatives(doc: dict) -> list[dict]:
    """Match financial tables with total assets and accounting-style parenthetical negatives."""
    import re
    try:
        out = []
        neg_re = re.compile(r"\(\d[\d,\.]*\)")
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = span.get("text") or ""
            if "total assets" in txt.lower() and neg_re.search(txt):
                out.append(span)
        return out
    except Exception:
        return []
