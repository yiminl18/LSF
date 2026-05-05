def rule_page1_cover_page_reference_date_labels(doc: dict) -> list[dict]:
    """Match label spans that define the cover-page reference date for outstanding shares."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") not in (1, 2):
                continue
            t = (span.get("text") or "").lower()
            if ("as of" in t and ("number of shares" in t or "outstanding" in t)) or ("issued and outstanding" in t):
                out.append(span)
        return out
    except Exception:
        return []
