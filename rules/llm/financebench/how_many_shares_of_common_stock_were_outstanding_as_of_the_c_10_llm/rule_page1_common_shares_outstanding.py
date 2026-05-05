def rule_page1_common_shares_outstanding(doc: dict) -> list[dict]:
    """Match spans with 'common shares outstanding' wording."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            t = " ".join((span.get("text") or "").lower().split())
            if span.get("page_no") in (1, 2):
                if re.search(r"common shares outstanding", t):
                    out.append(span)
        return out
    except Exception:
        return []
