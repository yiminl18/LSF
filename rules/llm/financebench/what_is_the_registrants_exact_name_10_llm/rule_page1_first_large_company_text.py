def rule_page1_first_large_company_text(doc: dict) -> list[dict]:
    """Match large page-1 text/section_header spans that look like a company name line."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            if (span.get("size") or 0) < 10:
                continue
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if "form 10-k" in low or "commission" in low:
                continue
            if "annual report" in low or "transition report" in low:
                continue
            if any(ch.isalpha() for ch in txt):
                out.append(span)
        return out
    except Exception:
        return []
