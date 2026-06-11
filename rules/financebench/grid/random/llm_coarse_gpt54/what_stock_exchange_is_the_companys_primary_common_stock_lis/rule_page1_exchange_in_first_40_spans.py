def rule_page1_exchange_in_first_40_spans(doc: dict) -> list[dict]:
    """Match exchange-like spans among the first 40 spans, reflecting the cover-page location pattern."""
    try:
        import re
        texts = doc.get("texts", [])[:40]
        out = []
        ex_re = re.compile(r"\b(new york stock exchange|the new york stock exchange|nasdaq|nasdaq global select market|the nasdaq global select market)\b")
        for span in texts:
            if ex_re.search((span.get("text") or "").lower()):
                out.append(span)
        return out
    except Exception:
        return []
