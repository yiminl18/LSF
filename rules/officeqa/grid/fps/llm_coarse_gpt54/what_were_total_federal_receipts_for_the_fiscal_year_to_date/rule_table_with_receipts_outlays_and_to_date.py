def rule_table_with_receipts_outlays_and_to_date(doc: dict) -> list[dict]:
    """Match tables containing receipts, outlays, and a fiscal-to-date row."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = span.get("text") or ""
            if (
                re.search(r'(Net receipts|Total receipts|Net budget receipts)', txt, re.I)
                and re.search(r'(Net outlays|Total outlays)', txt, re.I)
                and re.search(r'(Fiscal \d{4} to date|Actual fiscal year to date)', txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
