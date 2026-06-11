def rule_cover_page_first_company_after_commission_file(doc: dict) -> list[dict]:
    """Match the first company-like span on page 1 after a commission file number mention."""
    try:
        texts = doc.get("texts", [])
        seen_file = False
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            blob = ((span.get("text", "") or "") + " " + (span.get("text_span", "") or "")).lower()
            if "commission file number" in blob or "commission file no." in blob or "commission file no" in blob or "commission file number:" in blob:
                seen_file = True
                continue
            if not seen_file:
                continue
            txt = (span.get("text", "") or "").strip()
            if not txt:
                continue
            if span.get("bold", 0) == 1 and float(span.get("size", 0) or 0) >= 9:
                if "state or other jurisdiction" in blob or "address of principal executive offices" in blob:
                    continue
                if "form 10-" in blob or "form 8-k" in blob or "current report" in blob:
                    continue
                out.append(span)
                break
        return out
    except Exception:
        return []
