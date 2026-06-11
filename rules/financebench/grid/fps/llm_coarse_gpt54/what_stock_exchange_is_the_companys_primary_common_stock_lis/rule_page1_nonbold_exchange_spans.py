def rule_page1_nonbold_exchange_spans(doc: dict) -> list[dict]:
    """Match non-bold page-1 spans whose text is an exchange name."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and span.get("bold") == 0:
                txt = (span.get("text") or "").strip()
                if re.search(r'new york stock exchange|the nasdaq global select market|the nasdaq stock market llc|NASDAQ$', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
