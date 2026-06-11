def rule_page1_near_company_name_with_ein_or_state(doc: dict) -> list[dict]:
    """Match page-1 spans within 12 spans of the company name that look like state or EIN values."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        idx = None
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if any(k in txt.lower() for k in ["inc.", "incorporated", "corporation", "company", "plc"]):
                idx = i
                break
        if idx is None:
            return []
        for cand in texts[idx:idx+12]:
            if cand.get("page_no") != 1:
                continue
            c = (cand.get("text") or "").strip()
            if re.fullmatch(r"\d{2}-\d{7}", c):
                out.append(cand)
            elif re.fullmatch(r"(Delaware|New York|New Jersey|Washington|California|Minnesota|Jersey)", c, re.I):
                out.append(cand)
        return out
    except Exception:
        return []
