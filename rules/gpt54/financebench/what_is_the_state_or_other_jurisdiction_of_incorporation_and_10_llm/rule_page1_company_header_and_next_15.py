def rule_page1_company_header_and_next_15(doc: dict) -> list[dict]:
    """Match the first company-name header on page 1 and the next 15 spans after it."""
    try:
        texts = doc.get("texts", [])
        idx0 = None
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                txt = (span.get("text") or "").strip().lower()
                if txt and "commission" not in txt and "form 10-k" not in txt and "annual report" not in txt:
                    idx0 = i
                    break
        if idx0 is None:
            return []
        return [s for s in texts[idx0: min(len(texts), idx0 + 16)] if s.get("page_no") == 1]
    except Exception:
        return []
