def rule_page1_identification_keywords_any_label(doc: dict) -> list[dict]:
    """Match any page-1 span with identification keywords like incorporation, organization, or employer identification."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if "jurisdiction of incorporation" in txt or "employer identification" in txt or "incorporation or organization" in txt:
                out.append(span)
        return out
    except Exception:
        return []
