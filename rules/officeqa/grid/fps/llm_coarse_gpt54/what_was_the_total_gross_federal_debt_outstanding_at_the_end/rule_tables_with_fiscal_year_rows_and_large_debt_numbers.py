def rule_tables_with_fiscal_year_rows_and_large_debt_numbers(doc: dict) -> list[dict]:
    """Match tables with fiscal-year rows and many 6+ digit numbers, a broad recall rule for debt summary tables."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if not re.search(r'fiscal year|fiscal \d{4}|20\d{2}|19\d{2}', text, re.I):
                continue
            nums = re.findall(r'\b\d{6,}\b', text.replace(",", ""))
            if len(nums) >= 3:
                out.append(span)
    except Exception:
        return []
    return out
