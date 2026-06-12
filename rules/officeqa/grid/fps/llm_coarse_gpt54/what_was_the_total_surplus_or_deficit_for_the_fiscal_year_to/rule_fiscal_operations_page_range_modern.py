def rule_fiscal_operations_page_range_modern(doc: dict) -> list[dict]:
    """Match likely answer tables on modern early content pages (roughly pages 10-25) with fiscal summary wording."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            p = span.get("page_no")
            if span.get("label") == "table" and isinstance(p, int) and 8 <= p <= 25:
                txt = (span.get("text") or "")
                if re.search(r'(summary of fiscal operations|total surplus.*deficit|fiscal year to date)', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
