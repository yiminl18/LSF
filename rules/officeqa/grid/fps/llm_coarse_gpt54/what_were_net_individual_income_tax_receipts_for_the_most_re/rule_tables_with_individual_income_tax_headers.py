def rule_tables_with_individual_income_tax_headers(doc: dict) -> list[dict]:
    """Match tables containing individual income tax column headers such as Withheld/Other/Refunds/Net."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'Individual', txt, re.I) and re.search(r'Withheld', txt, re.I) and re.search(r'Refunds', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
