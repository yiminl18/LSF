def rule_page_range_late_financials(doc: dict) -> list[dict]:
    """Match later-document tables on pages 40+ that mention debt, useful for long 10-Ks."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table" and (span.get("page_no") or 0) >= 40:
                txt = (span.get("text") or "").lower()
                if re.search(r"\bdebt\b", txt) or re.search(r"\bborrowings\b", txt):
                    out.append(span)
        return out
    except Exception:
        return []
