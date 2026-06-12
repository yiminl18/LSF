def rule_receipts_table_with_fiscal_to_date(doc: dict) -> list[dict]:
    """Match receipt tables containing a fiscal-to-date row and a receipts column."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = span.get("text") or ""
            if (
                re.search(r'Fiscal\s+\d{4}\s+to\s+date', txt, re.I)
                and (
                    re.search(r'Net budget receipts', txt, re.I)
                    or re.search(r'Net receipts', txt, re.I)
                    or re.search(r'Total receipts', txt, re.I)
                )
            ):
                out.append(span)
        return out
    except Exception:
        return []
