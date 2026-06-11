def rule_page1_exchange_name_spans(doc: dict) -> list[dict]:
    """Match page-1 spans whose text looks like a stock exchange name."""
    try:
        import re
        out = []
        pats = [
            r"new york stock exchange",
            r"nasdaq(?: global select market)?",
            r"the nasdaq global select market",
            r"the nasdaq global select market",
            r"the nasdaq global select market",
            r"chicago stock exchange",
            r"australian securities exchange",
            r"the nasdaq global select market",
            r"the nasdaq global select market",
            r"the nasdaq global select market",
        ]
        for span in doc.get("texts", []):
            txt = span.get("text", "") or ""
            if span.get("page_no") == 1 and any(re.search(p, txt, re.I) for p in pats):
                out.append(span)
        return out
    except Exception:
        return []
