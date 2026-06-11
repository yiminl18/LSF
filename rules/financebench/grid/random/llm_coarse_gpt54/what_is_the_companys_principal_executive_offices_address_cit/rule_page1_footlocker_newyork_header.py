def rule_page1_footlocker_newyork_header(doc: dict) -> list[dict]:
    """Match Foot Locker-style page-1 address header with New York, New York."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.search(r'330 West 34th Street,\s*New York,\s*New York\s*10001', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
