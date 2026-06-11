def rule_page1_form_and_company_cover_cluster(doc: dict) -> list[dict]:
    """Match page-1 cover cluster spans between FORM header and company name if they contain reporting-period clues."""
    try:
        texts = doc.get("texts", [])
        form_idx = None
        company_idx = None
        for i, s in enumerate(texts):
            t = (s.get("text") or "").lower()
            if form_idx is None and s.get("page_no") == 1 and t.startswith("form "):
                form_idx = i
            if form_idx is not None and company_idx is None and s.get("page_no") == 1 and any(x in t for x in [
                "inc.", "corporation", "company", "plc", "incorporated"
            ]):
                company_idx = i
                break
        if form_idx is None:
            return []
        end = company_idx if company_idx is not None else min(len(texts), form_idx + 12)
        out = []
        for s in texts[form_idx:end]:
            t = (s.get("text") or "").lower()
            if any(k in t for k in ["fiscal year ended", "quarterly period ended", "date of report", "transition period"]):
                out.append(s)
        return out
    except Exception:
        return []
