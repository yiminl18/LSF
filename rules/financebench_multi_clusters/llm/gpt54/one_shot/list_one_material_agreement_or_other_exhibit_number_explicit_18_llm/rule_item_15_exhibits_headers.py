def rule_item_15_exhibits_headers(doc: dict) -> list[dict]:
    """Match headers mentioning Item 15 and Exhibits/Exhibit Index."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") not in {"section_header", "text"}:
                continue
            txt = (span.get("text") or "")
            if re.search(r"\bitem\s*15\b", txt, re.I) and re.search(r"\bexhibit", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
