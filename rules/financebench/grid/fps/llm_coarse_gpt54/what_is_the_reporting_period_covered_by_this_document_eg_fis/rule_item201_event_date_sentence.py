def rule_item201_event_date_sentence(doc: dict) -> list[dict]:
    """Match spans under Item 2.01 that begin with an event date sentence."""
    import re
    try:
        out = []
        pat = re.compile(r'^\s*on\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s+\d{4}', re.I)
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            text = span.get("text") or ""
            if "item 2.01" in path and pat.search(text):
                out.append(span)
        return out
    except Exception:
        return []
