def rule_page1_exchange_in_text_span(doc: dict) -> list[dict]:
    """Match page-1 spans whose text_span field contains exchange names."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        pats = [
            r"\bnew york stock exchange\b",
            r"\bthe new york stock exchange\b",
            r"\bnasdaq\b",
            r"\bnasdaq global select market\b",
            r"\bthe nasdaq global select market\b",
        ]
        for span in texts:
            if span.get("page_no") == 1:
                low = (span.get("text_span") or "").lower()
                if any(re.search(p, low) for p in pats):
                    out.append(span)
        return out
    except Exception:
        return []
