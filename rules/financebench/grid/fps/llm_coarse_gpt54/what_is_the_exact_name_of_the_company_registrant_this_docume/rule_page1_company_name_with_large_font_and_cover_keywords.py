def rule_page1_company_name_with_large_font_and_cover_keywords(doc: dict) -> list[dict]:
    """Match page-1 large-font spans whose nearby context includes cover-page registrant keywords."""
    try:
        texts = doc.get("texts", [])
        out = []
        keys = ["registrant", "commission file", "employer identification", "principal executive offices", "telephone number"]
        for i, span in enumerate(texts):
            if span.get("page_no") != 1 or float(span.get("size", 0) or 0) < 10:
                continue
            txt = (span.get("text", "") or "").strip()
            if not txt:
                continue
            blob = " ".join((((w.get("text", "") or "") + " " + (w.get("text_span", "") or "")) for w in texts[max(0, i-2):min(len(texts), i+12)])).lower()
            if sum(k in blob for k in keys) >= 2:
                if "form 10-" not in txt.lower() and "form 8-k" not in txt.lower():
                    out.append(span)
        return out
    except Exception:
        return []
