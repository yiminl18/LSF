def rule_page1_near_company_name_cover_block(doc: dict) -> list[dict]:
    """Match page-1 outstanding-share spans occurring under a company-name cover block rather than Item sections."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            path = ((span.get("structure") or {}).get("path_text") or "")
            t = (span.get("text") or "").lower()
            if "item 1" in path.lower():
                continue
            if "outstanding" in t and ("common stock" in t or "shares" in t):
                out.append(span)
    except Exception:
        return []
    return out
