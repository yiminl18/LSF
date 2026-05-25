def rule_consolidated_balance_sheet_total_assets(doc: dict) -> list[dict]:
    '''Audited Consolidated Balance Sheet table containing "Total assets", plus any short "(in millions)" / section-header context on the same page.'''
    texts = doc.get("texts", [])
    bs_spans = []
    bs_pages = set()
    for span in texts:
        if span.get("label") != "table":
            continue
        text = span.get("text") or ""
        lower = text.lower()
        if "total assets" not in lower:
            continue
        path_text = ((span.get("structure") or {}).get("path_text") or "").lower()
        if "consolidated balance sheet" in path_text or "consolidated balance sheet" in lower:
            bs_spans.append(span)
            page = span.get("page_no")
            if page is not None:
                bs_pages.add(page)
    extras = []
    for span in texts:
        if span.get("page_no") not in bs_pages:
            continue
        if span.get("label") == "table":
            continue
        text = (span.get("text") or "").strip()
        if not text or len(text) > 220:
            continue
        lower = text.lower()
        path_text = ((span.get("structure") or {}).get("path_text") or "").lower()
        if "in millions" in lower or "consolidated balance sheet" in lower or "consolidated balance sheet" in path_text or "fiscal year" in lower:
            extras.append(span)
    return bs_spans + extras
