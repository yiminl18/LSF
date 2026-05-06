def rule_page1_top_half_address_candidates(doc: dict) -> list[dict]:
    """Match likely address candidates in the top half of page 1."""
    import re
    try:
        texts = [s for s in doc.get("texts", []) if s.get("page_no") == 1]
        out = []
        limit = max(1, len(texts) // 2)
        for span in texts[:limit]:
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r'\d{1,5}\s', text) and (
                re.search(r'\b\d{5}(?:-\d{4})?\b', text) or
                re.search(r'United Kingdom', text, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
