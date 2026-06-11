def rule_item1_overview_executive_offices(doc: dict) -> list[dict]:
    """Match overview/business spans that include the full executive office address sentence."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = span.get("text") or ""
            if re.search(r'overview|item 1|business', path, re.I) and re.search(r'executive offices.*located at', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
