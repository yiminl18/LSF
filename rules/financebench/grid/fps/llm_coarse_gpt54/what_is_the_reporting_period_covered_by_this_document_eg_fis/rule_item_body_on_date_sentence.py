def rule_item_body_on_date_sentence(doc: dict) -> list[dict]:
    """Match item-body sentences starting with 'On <date>' on pages 2-3, useful for event-date answers when cover date differs."""
    import re
    try:
        out = []
        pat = re.compile(r'^\s*(\([a-z]\)\s*)?on\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s+\d{4}', re.I)
        for span in doc.get("texts", []):
            if span.get("page_no") in [2, 3]:
                text = span.get("text") or ""
                if pat.search(text):
                    out.append(span)
        return out
    except Exception:
        return []
