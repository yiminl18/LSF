def rule_financial_operations_section_tables(doc: dict) -> list[dict]:
    """Match tables under Financial Operations / Federal Fiscal Operations paths with receipt-related content."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = span.get("text") or ""
            if (
                re.search(r'Financial Operations', path, re.I)
                and (
                    re.search(r'Federal Fiscal Operations', path, re.I)
                    or re.search(r'Summary of Fiscal Operations', path, re.I)
                    or re.search(r'Receipts by Source', path, re.I)
                )
                and re.search(r'(receipts|outlays|fiscal year to date)', txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
