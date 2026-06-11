def rule_long_term_liabilities_with_debt(doc: dict) -> list[dict]:
    """Match tables/spans where 'long-term liabilities' and 'debt' co-occur."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if re.search(r"long[\-\s]?term liabilities", text, re.I) and re.search(r"\bdebt\b", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
