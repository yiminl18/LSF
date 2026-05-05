def rule_page_range_mid_financials(doc: dict) -> list[dict]:
    """Match mid-document tables on pages 20-60 that mention debt, useful for shorter 10-Ks."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            p = span.get("page_no") or 0
            if span.get("label") == "table" and 20 <= p <= 60:
                txt = (span.get("text") or "").lower()
                if re.search(r"\bdebt\b", txt) or re.search(r"\bborrowings\b", txt):
                    out.append(span)
        return out
    except Exception:
        return []
