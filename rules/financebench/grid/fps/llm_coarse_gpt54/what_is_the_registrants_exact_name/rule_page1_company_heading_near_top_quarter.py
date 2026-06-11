def rule_page1_company_heading_near_top_quarter(doc: dict) -> list[dict]:
    """Match prominent spans in the first quarter of page-1 text order that look like company names."""
    try:
        texts = doc.get("texts", [])
        page1 = [s for s in texts if s.get("page_no") == 1]
        cutoff = max(1, len(page1) // 4)
        out = []
        for span in page1[:cutoff + 10]:
            txt = (span.get("text") or "").strip().lower()
            if span.get("bold") == 1 and float(span.get("size") or 0) >= 10:
                if "form 10-" not in txt and "form 8-k" not in txt and "securities and exchange commission" not in txt and "current report" not in txt:
                    out.append(span)
        return out
    except Exception:
        return []
