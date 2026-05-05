def rule_page1_registrants_common_stock_outstanding(doc: dict) -> list[dict]:
    """Match spans with 'registrant's common stock outstanding' wording."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            t = " ".join((span.get("text") or "").lower().split())
            if span.get("page_no") in (1, 2):
                if re.search(r"registrant[’']?s common stock .* outstanding", t):
                    out.append(span)
        return out
    except Exception:
        return []
