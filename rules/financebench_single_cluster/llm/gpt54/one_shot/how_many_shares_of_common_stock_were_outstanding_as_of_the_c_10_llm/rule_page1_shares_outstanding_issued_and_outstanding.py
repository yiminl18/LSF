def rule_page1_shares_outstanding_issued_and_outstanding(doc: dict) -> list[dict]:
    """Match spans with 'shares of common stock issued and outstanding as of' wording."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            t = " ".join((span.get("text") or "").lower().split())
            if span.get("page_no") in (1, 2):
                if re.search(r"shares of common stock issued and outstanding as of", t):
                    out.append(span)
        return out
    except Exception:
        return []
