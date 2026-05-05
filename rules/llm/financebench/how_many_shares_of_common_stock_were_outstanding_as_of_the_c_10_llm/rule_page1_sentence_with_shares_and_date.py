def rule_page1_sentence_with_shares_and_date(doc: dict) -> list[dict]:
    """Match page-1 spans containing shares plus a date phrase likely tied to the cover-page reference date."""
    out = []
    try:
        months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
        for span in doc.get("texts", []):
            t = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "shares" in t and "outstanding" in t and any(m in t for m in months):
                out.append(span)
    except Exception:
        return []
    return out
