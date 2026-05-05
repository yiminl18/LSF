def rule_financial_statement_table_with_parentheses_and_dollars(doc: dict) -> list[dict]:
    """Match likely financial statement tables with dollar formatting and a net income row."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = span.get("text", "") or ""
            if not re.search(r"\bnet income\b|\bnet earnings\b|\bnet loss\b", txt, re.I):
                continue
            if re.search(r"\$|\(|\)|million|billion", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
