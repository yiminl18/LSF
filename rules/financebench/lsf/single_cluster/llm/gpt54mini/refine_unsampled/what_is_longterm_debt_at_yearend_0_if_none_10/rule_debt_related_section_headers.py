def rule_debt_related_section_headers(doc: dict) -> list[dict]:
    """Match debt-related headers and nearby text spans as broad anchors."""
    import re
    try:
        out = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("label") == "section_header":
                txt = (span.get("text") or "").lower()
                if re.search(r"\bdebt\b", txt) or re.search(r"\bborrowings\b", txt):
                    out.append(span)
                    for j in range(i + 1, min(i + 4, len(texts))):
                        if texts[j].get("page_no") == span.get("page_no"):
                            out.append(texts[j])
        return out
    except Exception:
        return []
