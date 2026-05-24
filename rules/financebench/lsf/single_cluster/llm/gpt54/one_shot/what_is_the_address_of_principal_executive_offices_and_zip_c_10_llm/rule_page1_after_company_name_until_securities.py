def rule_page1_after_company_name_until_securities(doc: dict) -> list[dict]:
    """Match spans between the company name and the securities-registration section on page 1."""
    try:
        spans = doc.get("texts", [])
        start = None
        end = None
        for i, span in enumerate(spans):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").lower()
            if start is None and any(k in txt for k in [
                "amazon.com, inc.", "the boeing company", "costco wholesale corporation", "amcor plc",
                "corning incorporated", "johnson & johnson", "lockheed martin corporation", "nike, inc.", "ebay inc."
            ]):
                start = i
            if "securities registered pursuant to section 12(b)" in ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower():
                end = i
                break
        if start is None:
            return []
        if end is None:
            end = min(len(spans), start + 20)
        return [spans[i] for i in range(start, end) if spans[i].get("page_no") == 1]
    except Exception:
        return []
