def rule_page1_company_heading_excluding_item_headers(doc: dict) -> list[dict]:
    """Match page-1 prominent spans excluding item headers and later content."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = (span.get("text") or "").strip().lower()
            if span.get("page_no") != 1:
                continue
            if "item " in txt or "part i" in txt:
                continue
            if span.get("bold") == 1 and float(span.get("size") or 0) >= 10:
                if "form 10-" not in txt and "form 8-k" not in txt and "securities and exchange commission" not in txt:
                    out.append(span)
        return out
    except Exception:
        return []
