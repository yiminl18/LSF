def rule_tables_with_individual_income_taxes_and_numeric_values(doc: dict) -> list[dict]:
    """Match tables mentioning individual income taxes and containing decimal/numeric values likely to include the answer."""
    import re
    out = []
    try:
        num_re = re.compile(r'\b\d+(?:\.\d+)?\b')
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'Individual income taxes|Individual', txt, re.I) and len(num_re.findall(txt)) >= 6:
                out.append(span)
    except Exception:
        return []
    return out
