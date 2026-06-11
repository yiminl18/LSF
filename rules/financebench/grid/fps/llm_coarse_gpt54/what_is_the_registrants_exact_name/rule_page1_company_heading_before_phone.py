def rule_page1_company_heading_before_phone(doc: dict) -> list[dict]:
    """Match prominent page-1 spans before the registrant telephone line."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            low = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "telephone number" in low:
                for j in range(max(0, i - 8), i):
                    prev = texts[j]
                    ptxt = (prev.get("text") or "").lower()
                    if prev.get("page_no") == 1 and prev.get("bold") == 1 and float(prev.get("size") or 0) >= 10:
                        if "address of principal executive offices" not in ptxt and "i.r.s." not in ptxt and "commission file" not in ptxt:
                            out.append(prev)
                break
        return out
    except Exception:
        return []
