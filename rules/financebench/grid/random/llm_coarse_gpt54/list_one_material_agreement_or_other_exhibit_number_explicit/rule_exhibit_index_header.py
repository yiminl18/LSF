def rule_exhibit_index_header(doc: dict) -> list[dict]:
    """Match section headers or text spans explicitly labeled 'Exhibit Index'."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if re.search(r"\bexhibit index\b", txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
