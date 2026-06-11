def rule_item1_business_overview_address_sentence(doc: dict) -> list[dict]:
    """Match Business Overview/Overview text that states executive offices location."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "").strip()
            if re.search(r'Overview|Business Overview|OVERVIEW', path, re.I) and re.search(r'executive offices.*located at|principal corporate offices.*located in', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
