def rule_table_with_fiscal_year_or_month_and_receipts(doc: dict) -> list[dict]:
    """Match tables whose headers include Fiscal year or month and a receipts column."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = span.get("text") or ""
            if re.search(r'Fiscal year or month', txt, re.I) and re.search(r'(Net receipts|Total receipts|Net budget receipts)', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
