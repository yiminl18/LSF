def rule_page2_cover_page_numeric_sentence(doc: dict) -> list[dict]:
    """Match page-2 text spans with a long comma-formatted number and outstanding-share keywords for spillover cover pages."""
    import re
    out = []
    try:
        num_re = re.compile(r"\b\d{1,3}(?:,\d{3}){2,}\b")
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            t = text.lower()
            if span.get("page_no") == 2 and num_re.search(text):
                if "outstanding" in t and ("common stock" in t or "shares" in t):
                    out.append(span)
    except Exception:
        return []
    return out
