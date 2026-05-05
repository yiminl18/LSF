def rule_8k_exhibit_bylaws(doc: dict) -> list[dict]:
    """Match spans mentioning bylaws exhibits, especially Exhibit 3.2."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r"\b3\.2\b", txt) and re.search(r"\bbylaws\b", txt, re.I):
                out.append(span)
            elif re.search(r"\bbylaws\b", txt, re.I) and re.search(r"\bexhibit\b", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
