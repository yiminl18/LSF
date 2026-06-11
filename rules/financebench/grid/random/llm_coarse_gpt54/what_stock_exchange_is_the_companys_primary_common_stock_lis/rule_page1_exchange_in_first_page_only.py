def rule_page1_exchange_in_first_page_only(doc: dict) -> list[dict]:
    """High-recall page-1 exchange-name matcher for cover-page answers."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        ex_re = re.compile(r"\b(new york stock exchange|the new york stock exchange|nasdaq|nasdaq global select market|the nasdaq global select market|nyse)\b")
        for span in texts:
            if span.get("page_no") == 1 and ex_re.search((span.get("text") or "").lower()):
                out.append(span)
        return out
    except Exception:
        return []
