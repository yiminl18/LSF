def rule_exhibit_bold_text_lines(doc: dict) -> list[dict]:
    """Match bold spans that contain exhibit references or exhibit descriptions."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            if span.get("bold") == 1 and re.search(r"\b(exhibit|agreement|plan|indenture|bylaws|press release)\b", txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
