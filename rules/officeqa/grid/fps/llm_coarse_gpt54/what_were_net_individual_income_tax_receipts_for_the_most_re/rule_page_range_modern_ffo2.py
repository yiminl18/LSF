def rule_page_range_modern_ffo2(doc: dict) -> list[dict]:
    """Match likely answer tables on modern documents where FFO-2 usually appears in early teen pages."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            p = span.get("page_no")
            txt = (span.get("text") or "")
            if isinstance(p, int) and 15 <= p <= 25:
                if re.search(r'FFO-2|Budget Receipts by Source|On-Budget and Off-Budget Receipts by Source', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
