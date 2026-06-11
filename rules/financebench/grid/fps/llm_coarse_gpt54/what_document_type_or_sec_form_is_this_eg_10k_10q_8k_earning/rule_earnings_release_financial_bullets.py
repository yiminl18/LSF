def rule_earnings_release_financial_bullets(doc: dict) -> list[dict]:
    """Match page-1 bullet/list-item spans with earnings-release metrics like net sales, net earnings, free cash flow, or outlook."""
    import re
    try:
        out = []
        pat = r"(net sales|net earnings|free cash flow|financial outlook|returned .* cash to shareholders)"
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and span.get("label") in {"list_item", "section_header", "text"} and re.search(pat, text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
