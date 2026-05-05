def rule_page1_market_value_and_outstanding_same_span(doc: dict) -> list[dict]:
    """Match spans that contain both aggregate market value language and outstanding-share language."""
    try:
        out = []
        for span in doc.get("texts", []):
            t = (span.get("text") or "").lower()
            if span.get("page_no") in (1, 2):
                if ("aggregate market value" in t or "market value" in t) and "outstanding" in t:
                    out.append(span)
        return out
    except Exception:
        return []
