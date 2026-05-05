def rule_page1_after_company_name_before_phone(doc: dict) -> list[dict]:
    """Match spans between the company name block and the telephone line on page 1."""
    try:
        spans = doc.get("texts", [])
        phone_idx = None
        for i, s in enumerate(spans):
            if s.get("page_no") == 1:
                txt = ((s.get("text") or "") + " " + (s.get("text_span") or "")).lower()
                if "telephone number" in txt or "registrant’s telephone number" in txt or "registrant's telephone number" in txt:
                    phone_idx = i
                    break
        if phone_idx is None:
            return []
        start = max(0, phone_idx - 8)
        return [s for s in spans[start:phone_idx + 1] if s.get("page_no") == 1]
    except Exception:
        return []
