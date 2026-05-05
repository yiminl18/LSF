def rule_page1_shares_outstanding_as_of_october(doc: dict) -> list[dict]:
    """Match spans with outstanding-share sentence using an October reference date."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            t = " ".join((span.get("text") or "").lower().split())
            if span.get("page_no") in (1, 2):
                if "outstanding" in t and re.search(r"as of october \d{1,2}, \d{4}", t):
                    out.append(span)
        return out
    except Exception:
        return []
