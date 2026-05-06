def rule_page1_shares_of_common_stock_outstanding(doc: dict) -> list[dict]:
    """Match page-1 spans with 'shares of common stock outstanding' wording."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            if span.get("page_no") == 1:
                t = " ".join(text.lower().split())
                if re.search(r"shares of (our |the registrant[’']?s )?common stock.*outstanding", t):
                    out.append(span)
        return out
    except Exception:
        return []
