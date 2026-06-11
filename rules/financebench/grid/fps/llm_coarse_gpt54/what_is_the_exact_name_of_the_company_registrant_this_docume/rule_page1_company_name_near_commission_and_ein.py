def rule_page1_company_name_near_commission_and_ein(doc: dict) -> list[dict]:
    """Match page-1 span whose nearby context includes both commission file and EIN markers."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text", "") or "").strip()
            if not txt:
                continue
            window = texts[max(0, i-2):min(len(texts), i+12)]
            blob = " ".join(((w.get("text", "") or "") + " " + (w.get("text_span", "") or "")) for w in window).lower()
            if ("commission file" in blob or "commission file no" in blob) and ("employer identification no" in blob or "i.r.s. employer identification no" in blob or "irs employer identification no" in blob):
                if "form 10-" not in txt.lower() and "form 8-k" not in txt.lower():
                    out.append(span)
        return out
    except Exception:
        return []
