def rule_page1_company_heading_with_many_following_metadata(doc: dict) -> list[dict]:
    """Match page-1 prominent spans followed by many metadata lines like state, EIN, address, and phone."""
    try:
        texts = doc.get("texts", [])
        out = []
        keys = ["state or other jurisdiction", "i.r.s. employer identification", "irs employer identification", "address of principal executive offices", "telephone number"]
        for i, span in enumerate(texts):
            if span.get("page_no") != 1 or span.get("bold") != 1 or float(span.get("size") or 0) < 10:
                continue
            low = (span.get("text") or "").lower()
            if "form 10-" in low or "form 8-k" in low or "securities and exchange commission" in low:
                continue
            window = " ".join((s.get("text") or "").lower() for s in texts[i+1:i+12])
            score = sum(1 for k in keys if k in window)
            if score >= 2:
                out.append(span)
        return out
    except Exception:
        return []
