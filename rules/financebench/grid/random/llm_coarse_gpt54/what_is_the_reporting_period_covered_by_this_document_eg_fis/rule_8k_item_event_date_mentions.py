def rule_8k_item_event_date_mentions(doc: dict) -> list[dict]:
    """Match 8-K body spans that restate the event date with 'On <date>' near the top."""
    import re
    try:
        month = r'(January|February|March|April|May|June|July|August|September|October|November|December)'
        out = []
        for span in doc.get("texts", [])[:80]:
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'\bOn ' + month + r'\s+\d{1,2},\s+\d{4}\b', txt):
                out.append(span)
        return out
    except Exception:
        return []
