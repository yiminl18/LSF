def rule_any_table_with_long_term_debt_and_numeric_values(doc: dict) -> list[dict]:
    """Match any table containing long-term debt plus at least one numeric-looking value."""
    try:
        import re
        out = []
        num_pat = re.compile(r"\$?\s*\(?\d[\d,\.]*\)?")
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r"\blong[\-\s]?term debt\b", text, re.I) and num_pat.search(text):
                out.append(span)
        return out
    except Exception:
        return []
