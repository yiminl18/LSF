def rule_page1_cover_numeric_candidates(doc: dict) -> list[dict]:
    """Return large numeric page-1/2 spans as broad candidates for split-layout cover pages."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no") in (1, 2) and re.fullmatch(r"[\d,]{6,}", text):
                out.append(span)
        return out
    except Exception:
        return []
