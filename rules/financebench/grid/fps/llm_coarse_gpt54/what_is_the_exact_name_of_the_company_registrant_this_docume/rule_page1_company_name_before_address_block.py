def rule_page1_company_name_before_address_block(doc: dict) -> list[dict]:
    """Match page-1 company-like span that precedes the principal executive office address block."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text", "") or "").strip()
            if not txt:
                continue
            future = texts[i:i+10]
            if any("address of principal executive offices" in (((f.get("text", "") or "") + " " + (f.get("text_span", "") or "")).lower()) for f in future):
                if "form 10-" not in txt.lower() and "form 8-k" not in txt.lower() and "current report" not in txt.lower():
                    out.append(span)
        return out
    except Exception:
        return []
