def rule_8k_event_date_in_item_body(doc: dict) -> list[dict]:
    """Match 8-K item body spans on early pages that begin with 'On <Month> <day>, <year>'."""
    import re
    try:
        out = []
        pat = re.compile(r'^\s*(\([a-z]\)\s*)?on\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s+\d{4}', re.I)
        for span in doc.get("texts", []):
            text = span.get("text") or ""
            if span.get("page_no", 999) <= 3 and pat.search(text):
                path = ((span.get("structure") or {}).get("path_text") or "").lower()
                if "item " in path or "current report" in path or span.get("page_no") == 2:
                    out.append(span)
        return out
    except Exception:
        return []
