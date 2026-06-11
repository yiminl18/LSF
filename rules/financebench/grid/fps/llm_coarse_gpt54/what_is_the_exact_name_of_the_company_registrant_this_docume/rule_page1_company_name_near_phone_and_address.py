def rule_page1_company_name_near_phone_and_address(doc: dict) -> list[dict]:
    """Match page-1 company-like span with nearby phone/address cover details."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text", "") or "").strip()
            if not txt:
                continue
            low = txt.lower()
            if "form 10-" in low or "form 8-k" in low or "current report" in low:
                continue
            window = texts[i:i+15]
            blob = " ".join(((w.get("text", "") or "") + " " + (w.get("text_span", "") or "")) for w in window).lower()
            if "registrant" in blob and ("telephone number" in blob or "address of principal executive offices" in blob):
                if span.get("bold", 0) == 1 or float(span.get("size", 0) or 0) >= 10:
                    out.append(span)
        return out
    except Exception:
        return []
