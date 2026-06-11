def rule_item_15_exhibits_header(doc: dict) -> list[dict]:
    """Match spans mentioning Item 15 and Exhibits/Exhibit Index, a common parent location for the answer."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r"item\s*15", txt, re.I) and re.search(r"exhibit", txt, re.I):
                out.append(span)
            elif re.search(r"item\s*15", path, re.I) and re.search(r"exhibit", path, re.I):
                out.append(span)
    except Exception:
        return []
    return out
