def rule_page1_top_large_non_boilerplate(doc: dict) -> list[dict]:
    """Match top-of-page-1 large non-boilerplate spans among the first 20 spans."""
    try:
        texts = doc.get("texts", [])[:20]
        out = []
        for span in texts:
            txt = (span.get("text", "") or "").strip()
            low = txt.lower()
            if not txt or span.get("page_no") != 1:
                continue
            if any(x in low for x in ["united states", "securities and exchange commission", "washington", "form 10-k", "form 10-q", "form 8-k", "current report"]):
                continue
            if float(span.get("size", 0) or 0) >= 10:
                out.append(span)
        return out
    except Exception:
        return []
