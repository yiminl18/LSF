def rule_page1_address_with_state_name_and_zip(doc: dict) -> list[dict]:
    """Match page-1 spans containing state name plus ZIP code."""
    import re
    try:
        states = r'California|Washington|Minnesota|New York'
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = (span.get("text") or "").strip()
            if re.search(rf'\b({states})\s+\d{{5}}(?:-\d{{4}})?\b', text):
                out.append(span)
        return out
    except Exception:
        return []
