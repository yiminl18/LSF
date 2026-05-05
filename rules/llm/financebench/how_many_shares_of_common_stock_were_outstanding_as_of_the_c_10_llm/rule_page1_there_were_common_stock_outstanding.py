def rule_page1_there_were_common_stock_outstanding(doc: dict) -> list[dict]:
    """Match page-1 spans beginning with 'There were ... shares of common stock outstanding'."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            if span.get("page_no") in (1, 2):
                t = " ".join(text.lower().split())
                if re.search(r"there were .* shares .* common stock .* outstanding", t):
                    out.append(span)
        return out
    except Exception:
        return []
