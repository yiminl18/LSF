def rule_page1_symbol_like_allcaps_short(doc: dict) -> list[dict]:
    """Match short page-1 all-caps ticker-like spans (e.g., BA, ADBE, AMZN, COST, FL, ATVI, AMCR, MMM26)."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text", "") or "").strip()
            if span.get("page_no") != 1:
                continue
            if re.fullmatch(r"[A-Z]{1,5}(?:[/-][A-Z0-9]{1,5})?\d{0,2}", txt):
                out.append(span)
        return out
    except Exception:
        return []
