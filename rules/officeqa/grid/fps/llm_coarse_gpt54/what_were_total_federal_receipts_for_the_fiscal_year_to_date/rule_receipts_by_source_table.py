def rule_receipts_by_source_table(doc: dict) -> list[dict]:
    """Match FFO-2 / Budget Receipts by Source tables, which often contain the same fiscal-to-date receipts answer."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = span.get("text") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            if (
                re.search(r'FFO[-\s]?2', txt, re.I)
                or re.search(r'Budget Receipts by Source', txt, re.I)
                or re.search(r'On-?Budget and Off-?Budget Receipts by Source', txt, re.I)
                or re.search(r'FFO[-\s]?2', path, re.I)
                or re.search(r'Receipts by Source', path, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
