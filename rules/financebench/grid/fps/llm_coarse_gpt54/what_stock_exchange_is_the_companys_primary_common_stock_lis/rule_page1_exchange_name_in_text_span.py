def rule_page1_exchange_name_in_text_span(doc: dict) -> list[dict]:
    """Match page-1 spans whose text_span contains an exchange name."""
    import re
    try:
        pats = [
            r'new york stock exchange',
            r'the nasdaq global select market',
            r'the nasdaq stock market llc',
            r'\bnasdaq\b',
            r'\bnyse\b',
        ]
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text_span") or "").strip()
            if span.get("page_no") == 1 and any(re.search(p, txt, re.I) for p in pats):
                out.append(span)
        return out
    except Exception:
        return []
