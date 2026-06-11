def rule_page1_title_case_company_like(doc: dict) -> list[dict]:
    """Match title-case page-1 bold spans that look like company names and are near exact-name annotation."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text", "") or "").strip()
            if span.get("page_no") != 1 or not txt:
                continue
            if span.get("bold", 0) != 1:
                continue
            if "form 10-" in txt.lower() or "form 8-k" in txt.lower() or "current report" in txt.lower():
                continue
            if float(span.get("size", 0) or 0) < 9:
                continue
            window = texts[max(0, i-2):min(len(texts), i+5)]
            if any("exact name of registrant as specified in its charter" in ((w.get("text", "") or "") + " " + (w.get("text_span", "") or "")).lower() for w in window):
                out.append(span)
        return out
    except Exception:
        return []
