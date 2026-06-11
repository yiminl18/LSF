def rule_page1_company_name_before_address_block(doc: dict) -> list[dict]:
    """Match prominent page-1 company-name spans that occur before address/telephone blocks."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip().lower()
            if span.get("page_no") != 1:
                continue
            if span.get("bold") != 1 or float(span.get("size") or 0) < 10:
                continue
            if "form 10-" in txt or "form 8-k" in txt or "current report" in txt or "securities and exchange commission" in txt:
                continue
            window = texts[i+1:i+10]
            near_address = any(
                ("address of principal executive offices" in ((s.get("text") or "").lower()) or
                 "registrant" in ((s.get("text") or "").lower()) and "telephone" in ((s.get("text") or "").lower()) or
                 "i.r.s. employer identification" in ((s.get("text") or "").lower()) or
                 "irs employer identification" in ((s.get("text") or "").lower()))
                for s in window
            )
            if near_address:
                out.append(span)
        return out
    except Exception:
        return []
