def rule_row_total_receipts_with_actual_fytd(doc: dict) -> list[dict]:
    """Match summary tables containing both Total receipts and Actual fiscal year to date."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = span.get("text") or ""
                if re.search(r'Total receipts', txt, re.I) and re.search(r'Actual fiscal year to date', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
