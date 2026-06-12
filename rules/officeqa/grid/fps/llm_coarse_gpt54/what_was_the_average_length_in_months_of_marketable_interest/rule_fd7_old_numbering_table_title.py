def rule_fd7_old_numbering_table_title(doc: dict) -> list[dict]:
    """Match older table/title spans where the table is numbered FD-7/FO-7 and mentions maturity distribution and average length."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            low = txt.lower()
            if (
                re.search(r'\b(fd|fo)[-\s]?7\b', low)
                and "maturity distribution" in low
                and "average length" in low
            ):
                out.append(span)
    except Exception:
        return []
    return out
