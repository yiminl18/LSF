def rule_page1_exchange_name_spans(doc: dict) -> list[dict]:
    """Match page 1 spans whose text looks like an exchange name."""
    try:
        import re
        texts = doc.get("texts", [])
        pats = [
            r"new york stock exchange",
            r"\bnyse\b",
            r"nasdaq global select market",
            r"the nasdaq global select market",
            r"the new york stock exchange",
            r"australian securities exchange",
        ]
        out = []
        for span in texts:
            txt = (span.get("text") or "")
            if span.get("page_no") == 1 and any(re.search(p, txt, re.I) for p in pats):
                out.append(span)
        return out
    except Exception:
        return []
